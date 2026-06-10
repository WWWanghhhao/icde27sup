import torch
import torch.nn as nn
import torch.nn.functional as F


class TSRAG(nn.Module):
    def __init__(self, seq_len, feature_dim, d_model=64, top_k=3, dropout=0.1):
        """
        基于 TRACE 论文思路的 TS-to-TS 检索增强模块
        """
        super(TSRAG, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.d_model = d_model

        # 1. 轻量级 Encoder: 用于生成 Query Embedding (模拟 TRACE 的 [CLS] token)
        # 这里用 MLP 代替复杂的 Transformer 以适应轻量化需求，但逻辑一致
        self.query_encoder = nn.Sequential(
            nn.Linear(seq_len * feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. 软 Token 投影层 (Soft Token Projection) [cite: 796-798]
        # 将检索到的 Top-K 序列 (K * seq_len * feature_dim) 投影回原始维度或隐层维度
        # TRACE 是 concat 后投影，这里我们投影为 residual 增量
        self.rag_projector = nn.Sequential(
            nn.Linear(top_k * seq_len * feature_dim, seq_len * feature_dim),
            nn.LayerNorm(seq_len * feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层：将原始输入与检索增强特征融合
        self.fusion_gate = nn.Linear(seq_len * feature_dim * 2, seq_len * feature_dim)

        # Memory Bank (不参与梯度更新，作为外部知识库)
        # 存储格式: keys (embeddings), values (raw_scores)
        self.register_buffer("memory_keys", None)
        self.register_buffer("memory_values", None)

    def update_memory(self, batch_scores, strategy='score_priority'):
        """
        更新记忆库的高级实现。
        strategy:
            - 'fifo': 先进先出，保持固定大小 (标准做法)
            - 'score_priority': 优先保留具有高方差（信息量大/异常可能性大）的样本 (推荐)
        """
        B = batch_scores.shape[0]
        flat_input = torch.reshape(batch_scores, shape=[B, -1])

        # 编码得到 Keys
        with torch.no_grad():
            keys = self.query_encoder(flat_input)  # [B, d_model]

        if self.memory_keys is None:
            self.memory_keys = keys
            self.memory_values = batch_scores
            return

        new_keys = keys.detach()
        new_values = batch_scores.detach()

        updated_keys = torch.cat([self.memory_keys, new_keys], dim=0)
        updated_values = torch.cat([self.memory_values, new_values], dim=0)

        # 4. 裁剪机制 (Pruning)
        max_size = 2048  # 设定记忆库最大容量，根据显存调整
        current_size = updated_keys.shape[0]

        if current_size <= max_size:
            self.memory_keys = updated_keys
            self.memory_values = updated_values
        else:
            if strategy == 'fifo':
                # === 策略 A: 先进先出 (FIFO) ===
                # 只保留最近的 max_size 个
                self.memory_keys = updated_keys[-max_size:]
                self.memory_values = updated_values[-max_size:]

            elif strategy == 'score_priority':
                # === 策略 B: 异常分数优先级 (Score Priority) ===
                # 假设输入的 batch_scores 本身就是异常分数或重构误差
                # 计算每个样本的“重要性”。这里使用序列的方差或均值。
                # 方差大说明波动大，通常包含更多异常信息。

                # 计算重要性分数: [Current_Size]
                # 这里取每个样本所有通道和时间步的平均幅值作为重要性
                importance_scores = torch.mean(torch.abs(updated_values), dim=(1, 2))

                # 获取分数最高的 Top-K 个样本的索引
                _, top_indices = torch.topk(importance_scores, k=max_size)

                # 重新筛选，只保留最重要的样本
                self.memory_keys = updated_keys[top_indices]
                self.memory_values = updated_values[top_indices]

    def forward(self, x):
        """
        x: 当前输入的异常分数序列 [Batch, Seq_Len, Dim]
        """
        B, T, C = x.shape
        flat_x = torch.reshape(x, shape=[B, -1])

        # 1. 生成 Query Embedding
        query_emb = self.query_encoder(flat_x)  # [B, d_model]

        # 如果记忆库为空（第一轮），直接返回原始输入或零增强
        if self.memory_keys is None or self.memory_keys.shape[0] < self.top_k:
            return x

        # 2. 检索 (Retrieval) - 计算 Cosine Similarity
        # [B, d_model] @ [Mem_Size, d_model].T -> [B, Mem_Size]
        # TRACE 使用余弦相似度进行检索
        norm_query = F.normalize(query_emb, p=2, dim=1)
        norm_keys = F.normalize(self.memory_keys, p=2, dim=1)
        sim_matrix = torch.mm(norm_query, norm_keys.t())

        # 训练时要这笔掉自己 (Mask self-retrieval)，防止检索到完全一样的自己导致过拟合
        # 简单处理：如果是在训练且 memory 包含当前 batch，将对角线设为极小值
        # 这里假设 memory 是动态更新的，暂略去复杂 mask，直接取 Top-K

        _, indices = torch.topk(sim_matrix, k=self.top_k, dim=1)  # [B, K]

        # 3. 获取 Values (Raw Anomaly Scores)
        # indices: [B, K] -> retrieved: [B, K, T, C]
        retrieved_values = self.memory_values[indices]

        # 4. 生成 Soft Token / Prompt [cite: 796-797]
        # Flatten retrieved items: [B, K * T * C]
        # flat_retrieved = retrieved_values.view(B, -1)
        flat_retrieved = torch.reshape(retrieved_values, shape=[B, -1])

        # Project: [B, T * C]
        rag_features = self.rag_projector(flat_retrieved)

        # 5. 融合 (Augmentation)
        combined = torch.cat([flat_x, rag_features], dim=1)
        enhanced_x = self.fusion_gate(combined)  # [B, T*C]

        return torch.reshape(enhanced_x, shape=[B, T, C])


class OutputFusionRAG(nn.Module):
    def __init__(self, seq_len, pred_len, feature_dim, d_model=64, top_k=3):
        """
        Args:
            seq_len: 输入序列长度 (Look-back)
            pred_len: 预测序列长度 (Look-forward) -> 这是检索直接输出的目标维度
            feature_dim: 特征维度
            d_model: Query Encoder 的隐层维度
            top_k: 检索数量
        """
        super(OutputFusionRAG, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.d_model = d_model
        self.max_size = 128

        # 1. Query Encoder: 对 Look-back 进行编码生成检索用的 Key
        self.query_encoder = nn.Sequential(
            nn.Linear(seq_len * feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. 融合门控 (Adaptive Fusion Gate)
        # 输入: [Model_Pred, RAG_Pred, Input_Context] 的拼接
        # 输出: 融合权重 alpha (0~1)
        gate_input_dim = (pred_len * feature_dim * 2) + d_model
        self.fusion_net = nn.Sequential(
            nn.Linear(gate_input_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, feature_dim),  # 为每个通道生成独立的权重
            nn.Sigmoid()
        )

        # Memory Bank
        self.register_buffer("memory_keys", None)
        self.register_buffer("memory_values", None)

    def compute_contrastive_loss(self, query_emb, retrieved_keys, temperature=1):
        """
        计算 InfoNCE 对比损失
        目的：拉近 Query 与 Top-1 检索结果的距离，推远 Batch 内其他样本的检索结果

        Args:
            query_emb: [B, D]
            retrieved_keys: [B, K, D]
        """
        # 取检索到的 Top-1 Key 作为正样本 (Positive)
        # 假设检索排序第一的是最相关的历史片段
        pos_k = retrieved_keys[:, 0, :]  # [B, D]

        # 归一化 (Cosine Similarity 需要)
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_k = F.normalize(pos_k, p=2, dim=1)

        # Logits: [B, B]
        # 计算当前 Batch 中每个 Query 与 所有其他样本检索到的 Positive 的相似度
        # 对角线元素是 Positive Pair (Query_i, Key_i_Top1)
        logits = torch.mm(query_emb, pos_k.t()) / temperature

        # 生成标签 (对角线为 True Label)
        labels = torch.arange(query_emb.size(0)).to(query_emb.device)

        # Cross Entropy Loss
        loss = F.cross_entropy(logits, labels)
        return loss

    def update_memory(self, look_back_seq, look_forward_seq, strategy='score_priority'):
        """
        更新记忆库
        """
        B = look_back_seq.shape[0]
        flat_look_back = torch.reshape(look_back_seq, shape=[B, -1])

        with torch.no_grad():
            keys = self.query_encoder(flat_look_back)

        values = look_forward_seq.detach()

        if self.memory_keys is None:
            self.memory_keys = keys
            self.memory_values = values
            return

        # 拼接
        updated_keys = torch.cat([self.memory_keys, keys.detach()], dim=0)
        updated_values = torch.cat([self.memory_values, values], dim=0)

        # 裁剪 (Pruning)
        max_size = self.max_size
        current_size = updated_keys.shape[0]

        if current_size <= max_size:
            self.memory_keys = updated_keys
            self.memory_values = updated_values
        else:
            if strategy == 'fifo':
                self.memory_keys = updated_keys[-max_size:]
                self.memory_values = updated_values[-max_size:]
            elif strategy == 'score_priority':
                # 优先保留波动大的未来片段
                importance = torch.mean(torch.abs(updated_values), dim=(1, 2))
                _, top_indices = torch.topk(importance, k=max_size)
                self.memory_keys = updated_keys[top_indices]
                self.memory_values = updated_values[top_indices]

    def retrieve_and_aggregate(self, query_emb):
        """
        检索并聚合历史未来
        Returns:
            rag_prediction: [B, pred_len, C]
            retrieved_keys: [B, K, d_model] (用于计算 Loss)
        """
        # 1. 计算相似度
        norm_query = F.normalize(query_emb, p=2, dim=1)
        norm_keys = F.normalize(self.memory_keys, p=2, dim=1)
        sim_matrix = torch.mm(norm_query, norm_keys.t())  # [B, Mem_Size]

        # 训练时 Mask 掉自身 (防止检索泄露)
        if self.training:
            sim_matrix.masked_fill_(sim_matrix > 0.999, -1e9)

        # 2. Top-K 检索
        scores, indices = torch.topk(sim_matrix, k=self.top_k, dim=1)

        # 3. 获取检索内容
        # [B, K, pred_len, C]
        retrieved_values = self.memory_values[indices]
        # [B, K, d_model] - 获取 Keys 用于 Contrastive Loss
        retrieved_keys = self.memory_keys[indices]

        # 4. 基于相似度的加权聚合 (Softmax Aggregation)
        attn_weights = F.softmax(scores, dim=1)  # [B, K]

        # 扩充权重维度: [B, K, 1, 1]
        attn_weights = attn_weights.view(-1, self.top_k, 1, 1)

        # 加权求和: Sum(Weight_i * Value_i) -> [B, pred_len, C]
        rag_prediction = torch.sum(attn_weights * retrieved_values, dim=1)

        return rag_prediction, retrieved_keys

    def forward(self, x, model_prediction):
        """
        Args:
            x: 当前输入的 Look-back 序列 [B, seq_len, C]
            model_prediction: 预测器 (GWNet) 的原始输出 [B, pred_len, C]
        Returns:
            final_prediction: 融合后的输出 [B, pred_len, C]
            cl_loss: 对比损失 (Scalar)
        """
        B = x.shape[0]
        flat_x = torch.reshape(x, shape=[B, -1])

        # 1. 编码 Query
        query_emb = self.query_encoder(flat_x)  # [B, d_model]

        # 边界情况：记忆库不足
        if self.memory_keys is None or self.memory_keys.shape[0] < self.top_k:
            # 返回 0 loss
            return model_prediction, torch.tensor(0.0, device=x.device)

        # 2. 获取 RAG 预测与检索 Keys
        rag_prediction, retrieved_keys = self.retrieve_and_aggregate(query_emb)

        # 3. 计算对比损失
        cl_loss = self.compute_contrastive_loss(query_emb, retrieved_keys)

        # 4. 计算融合权重 (Gating)
        flat_model_pred = torch.reshape(model_prediction, shape=[B, -1])
        flat_rag_pred = torch.reshape(rag_prediction, shape=[B, -1])

        gate_input = torch.cat([flat_model_pred, flat_rag_pred, query_emb], dim=1)

        # alpha: [B, 1, C]
        alpha = self.fusion_net(gate_input).unsqueeze(1)

        # 5. 加权融合
        final_prediction = alpha * rag_prediction + (1 - alpha) * model_prediction

        return final_prediction, cl_loss





class OutputFusionRAG2(nn.Module):
    def __init__(self, seq_len, pred_len, feature_dim, d_model=64, top_k=3,
                 attn_temperature=0.5, gate_scale_init=1.0, max_size=64):
        """
        Args:
            seq_len: 输入序列长度 (Look-back)
            pred_len: 预测序列长度 (Look-forward)
            feature_dim: 特征维度
            d_model: Query Encoder 的隐层维度
            top_k: 检索数量
            attn_temperature: softmax温度(越小越尖锐)
            gate_scale_init: 融合放大因子初值
        """
        super(OutputFusionRAG2, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.d_model = d_model
        self.max_size = max_size
        self.attn_temperature = attn_temperature

        # 1. Query Encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(seq_len * feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. 融合门控（可放大）
        gate_input_dim = (pred_len * feature_dim)# + d_model
        self.fusion_net = nn.Sequential(
            nn.Linear(gate_input_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, feature_dim),
            nn.Tanh()
        )

        self.gate_scale = nn.Parameter(torch.ones(1, 1, feature_dim) * gate_scale_init)

        # Memory Bank
        self.register_buffer("memory_keys", None)
        self.register_buffer("memory_values", None)

    def compute_contrastive_loss(self, query_emb, retrieved_keys, temperature=1):
        pos_k = retrieved_keys[:, 0, :]
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_k = F.normalize(pos_k, p=2, dim=1)
        logits = torch.mm(query_emb, pos_k.t()) / temperature
        labels = torch.arange(query_emb.size(0)).to(query_emb.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def update_memory(self, look_back_seq, look_forward_seq, strategy='score_priority'):
        B = look_back_seq.shape[0]
        flat_look_back = torch.reshape(look_back_seq, shape=[B, -1])

        with torch.no_grad():
            keys = self.query_encoder(flat_look_back)

        values = look_forward_seq.detach()

        if self.memory_keys is None:
            self.memory_keys = keys
            self.memory_values = values
            return

        updated_keys = torch.cat([self.memory_keys, keys.detach()], dim=0)
        updated_values = torch.cat([self.memory_values, values], dim=0)

        max_size = self.max_size
        current_size = updated_keys.shape[0]

        if current_size <= max_size:
            self.memory_keys = updated_keys
            self.memory_values = updated_values
        else:
            if strategy == 'fifo':
                self.memory_keys = updated_keys[-max_size:]
                self.memory_values = updated_values[-max_size:]
            elif strategy == 'score_priority':
                importance = torch.mean(torch.abs(updated_values), dim=(1, 2))
                _, top_indices = torch.topk(importance, k=max_size)
                self.memory_keys = updated_keys[top_indices]
                self.memory_values = updated_values[top_indices]

    def retrieve_and_aggregate(self, query_emb):
        norm_query = F.normalize(query_emb, p=2, dim=1)
        norm_keys = F.normalize(self.memory_keys, p=2, dim=1)
        sim_matrix = torch.mm(norm_query, norm_keys.t())  # [B, Mem_Size]

        if self.training:
            sim_matrix.masked_fill_(sim_matrix > 0.999, -1e9)

        scores, indices = torch.topk(sim_matrix, k=self.top_k, dim=1)
        retrieved_values = self.memory_values[indices]
        retrieved_keys = self.memory_keys[indices]

        # 温度控制 softmax
        attn_weights = F.softmax(scores / self.attn_temperature, dim=1)
        attn_weights = attn_weights.view(-1, self.top_k, 1, 1)

        rag_prediction = torch.sum(attn_weights * retrieved_values, dim=1)

        return rag_prediction, retrieved_keys

    def forward(self, x, model_prediction):
        B = x.shape[0]
        flat_x = torch.reshape(x, shape=[B, -1])

        query_emb = self.query_encoder(flat_x)

        if self.memory_keys is None or self.memory_keys.shape[0] < self.top_k:
            return model_prediction, torch.tensor(0.0, device=x.device)

        rag_prediction, retrieved_keys = self.retrieve_and_aggregate(query_emb)
        cl_loss = self.compute_contrastive_loss(query_emb, retrieved_keys)


        flat_model_pred = torch.reshape(model_prediction, shape=[B, -1])
        flat_rag_pred = torch.reshape(rag_prediction, shape=[B, -1])
        gate_input = flat_rag_pred + flat_model_pred

        # alpha in [-1,1]
        alpha = self.fusion_net(gate_input).unsqueeze(1)
        alpha = (alpha * self.gate_scale).clamp(-2, 2)

        final_prediction = model_prediction + alpha * (rag_prediction - model_prediction)

        return final_prediction, cl_loss


class RAG3(nn.Module):
    """
    改进版 RAG：
    1) Query编码：先做每个channel的时间编码，再展平做全局编码
    2) 检索支持 cosine / mae / mse
    3) 对比损失改为：
       - 正样本：同批次内位置相邻（利用 dataloader shuffle=False 的顺序性）
       - 负样本：同批次内最远（与query最不相似）
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        feature_dim,
        d_model=64,
        top_k=3,
        attn_temperature=0.5,
        gate_scale_init=1.0,
        max_size=64,
        retrieve_metric='cosine',   # 'cosine' | 'mae' | 'mse'
        channel_embed_dim=8,
        cl_temperature=0.2
    ):
        super(RAG3, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.d_model = d_model
        self.max_size = max_size
        self.attn_temperature = attn_temperature
        self.retrieve_metric = retrieve_metric
        self.cl_temperature = cl_temperature

        # -------- Query Encoder: per-channel -> flatten -> global --------
        self.channel_encoder = nn.Sequential(
            nn.Linear(seq_len, max(16, d_model // 2)),
            nn.ReLU(),
            nn.Linear(max(16, d_model // 2), channel_embed_dim),
            nn.ReLU()
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(feature_dim * channel_embed_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 额外上下文编码（用于融合门控）
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # -------- 融合网络 --------
        fusion_in_dim = pred_len * feature_dim * 3 + d_model
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feature_dim),
            nn.Tanh()
        )
        self.gate_scale = nn.Parameter(torch.ones(1, 1, feature_dim) * gate_scale_init)

        # -------- Memory --------
        self.register_buffer("memory_keys", None)      # [M, d_model]
        self.register_buffer("memory_values", None)    # [M, pred_len, feature_dim]

    # =========================
    # 内部编码工具
    # =========================
    def _encode_per_channel_then_flatten(self, seq_3d):
        """
        seq_3d: [B, T, C]
        先按channel编码每条长度T，再展平后全局编码。
        """
        B, T, C = seq_3d.shape
        x = seq_3d.permute(0, 2, 1).contiguous()       # [B, C, T]
        x = x.view(B * C, T)                           # [B*C, T]
        ch_emb = self.channel_encoder(x)               # [B*C, E]
        ch_emb = ch_emb.view(B, C * ch_emb.shape[-1]) # [B, C*E]
        q = self.query_encoder(ch_emb)                 # [B, d_model]
        return q

    # =========================
    # 对比损失（改版）
    # =========================
    def compute_contrastive_loss(self, query_emb, positive_emb):
        """
        使用“同批次相邻样本”为正样本，“同批次最远样本”为负样本。
        - query_emb:    [B, d_model]
        - positive_emb: [B, d_model]  (由forward里按相邻位置构造)

        负样本选择：
        在 batch 内寻找与 query 最不相似（余弦最小）的样本作为负样本。
        """
        B = query_emb.size(0)
        if B < 2:
            return torch.tensor(0.0, device=query_emb.device)

        q = F.normalize(query_emb, p=2, dim=1)         # [B, d]
        p = F.normalize(positive_emb, p=2, dim=1)      # [B, d]

        # 正样本相似度
        sim_pos = torch.sum(q * p, dim=1)              # [B]

        # 负样本：batch内与q最不相似者
        sim_mat = torch.mm(q, q.t())                   # [B, B]
        sim_mat.fill_diagonal_(1e9)                    # 排除自己
        neg_idx = torch.argmin(sim_mat, dim=1)         # [B]
        n = q[neg_idx]                                 # [B, d]
        sim_neg = torch.sum(q * n, dim=1)              # [B]

        # 排序损失：希望 sim_pos > sim_neg
        loss = F.softplus((sim_neg - sim_pos) / max(self.cl_temperature, 1e-6)).mean()
        return loss

    # =========================
    # Memory更新
    # =========================
    def update_memory(self, look_back_seq, look_forward_seq, strategy='score_priority'):
        """
        look_back_seq: [B, seq_len, C]
        look_forward_seq: [B, pred_len, C]
        """
        with torch.no_grad():
            keys = self._encode_per_channel_then_flatten(look_back_seq)  # [B, d_model]
        values = look_forward_seq.detach()                                 # [B, pred_len, C]

        if self.memory_keys is None:
            self.memory_keys = keys
            self.memory_values = values
            return

        updated_keys = torch.cat([self.memory_keys, keys.detach()], dim=0)
        updated_values = torch.cat([self.memory_values, values], dim=0)

        current_size = updated_keys.shape[0]
        max_size = self.max_size

        if current_size <= max_size:
            self.memory_keys = updated_keys
            self.memory_values = updated_values
            return

        if strategy == 'fifo':
            self.memory_keys = updated_keys[-max_size:]
            self.memory_values = updated_values[-max_size:]
        else:  # score_priority
            importance = torch.mean(torch.abs(updated_values), dim=(1, 2))
            _, top_idx = torch.topk(importance, k=max_size)
            self.memory_keys = updated_keys[top_idx]
            self.memory_values = updated_values[top_idx]

    # =========================
    # 检索与聚合
    # =========================
    def retrieve_and_aggregate(self, query_emb):
        """
        query_emb: [B, d_model]
        return:
          rag_prediction: [B, pred_len, C]
          retrieved_keys: [B, K, d_model]
        """
        if self.retrieve_metric == 'cosine':
            q = F.normalize(query_emb, p=2, dim=1)                # [B, d]
            k = F.normalize(self.memory_keys, p=2, dim=1)         # [M, d]
            score_matrix = torch.mm(q, k.t())                     # [B, M], 越大越相似
            if self.training:
                score_matrix = score_matrix.masked_fill(score_matrix > 0.999, -1e9)
            scores, indices = torch.topk(score_matrix, k=self.top_k, dim=1, largest=True)

        elif self.retrieve_metric in ['mae', 'mse']:
            q = query_emb.unsqueeze(1)                            # [B,1,d]
            k = self.memory_keys.unsqueeze(0)                     # [1,M,d]
            diff = q - k                                          # [B,M,d]

            if self.retrieve_metric == 'mae':
                dist = torch.mean(torch.abs(diff), dim=-1)        # [B,M]
            else:
                dist = torch.mean(diff * diff, dim=-1)            # [B,M]

            score_matrix = -dist                                  # 距离越小，score越大
            scores, indices = torch.topk(score_matrix, k=self.top_k, dim=1, largest=True)
        else:
            raise ValueError(f"Unsupported retrieve_metric: {self.retrieve_metric}")

        retrieved_values = self.memory_values[indices]            # [B,K,pred_len,C]
        retrieved_keys = self.memory_keys[indices]                # [B,K,d_model]

        attn_weights = F.softmax(scores / self.attn_temperature, dim=1)  # [B,K]
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)           # [B,K,1,1]

        rag_prediction = torch.sum(attn_weights * retrieved_values, dim=1)  # [B,pred_len,C]
        return rag_prediction, retrieved_keys

    # =========================
    # 前向
    # =========================
    def forward(self, x, model_prediction):
        """
        x: [B, seq_len, C]
        model_prediction: [B, pred_len, C]
        """
        B = x.shape[0]
        query_emb = self._encode_per_channel_then_flatten(x)  # [B, d_model]

        if self.memory_keys is None or self.memory_keys.shape[0] < self.top_k:
            return model_prediction, torch.tensor(0.0, device=x.device)

        rag_prediction, retrieved_keys = self.retrieve_and_aggregate(query_emb)

        # ===== 正样本构建（按同批次相邻位置）=====
        # 因为 dataloader shuffle=False，相邻位置样本时间上更接近
        if B >= 2:
            pos_emb = torch.empty_like(query_emb)
            # 对 0..B-2，用 i+1 作为正样本
            pos_emb[:-1] = query_emb[1:]
            # 对最后一个，用前一个作为正样本
            pos_emb[-1] = query_emb[-2]
            cl_loss = self.compute_contrastive_loss(query_emb, pos_emb)
        else:
            cl_loss = torch.tensor(0.0, device=x.device)

        # ===== 融合 =====
        flat_model = model_prediction.reshape(B, -1)
        flat_rag = rag_prediction.reshape(B, -1)
        flat_diff = (rag_prediction - model_prediction).reshape(B, -1)

        top1_key = retrieved_keys[:, 0, :]  # [B, d_model]
        ctx = self.context_encoder(torch.cat([query_emb, top1_key], dim=-1))  # [B, d_model]

        gate_input = torch.cat([flat_model, flat_rag, flat_diff, ctx], dim=-1)
        alpha = self.fusion_net(gate_input).unsqueeze(1)  # [B,1,C] in [-1,1]
        alpha = (alpha * self.gate_scale).clamp(-2, 2)

        final_prediction = model_prediction + alpha * (rag_prediction - model_prediction)
        return final_prediction, cl_loss