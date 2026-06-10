import numpy as np
import torch

class StreamingInference:
    """
    通用的流式处理，支持自定义模型推理逻辑，降低推理时的内存占用
    """

    def __init__(self, original_length, seq_len, n_features, dtype=np.float32):
        self.original_length = original_length
        self.seq_len = seq_len
        self.n_features = n_features

        self.results = np.zeros((original_length, n_features), dtype=dtype)
        self.counts = np.zeros((original_length, n_features), dtype=dtype)
        self.window_idx = 0

    def add_window(self, window_data):
        """
        添加一个窗口的重构结果

        Args:
            window_data: numpy array, shape [seq_len, n_features] 或 [seq_len]
        """
        if window_data.ndim == 1:
            window_data = window_data[:, None]  # 转为 2D

        start_idx = self.window_idx
        end_idx = start_idx + self.seq_len

        if end_idx <= self.original_length:
            self.results[start_idx:end_idx] += window_data
            self.counts[start_idx:end_idx] += 1

        self.window_idx += 1

    def add_batch(self, batch_data):
        """
        添加一个 batch 的重构结果

        Args:
            batch_data: numpy array, shape [batch_size, seq_len, n_features]
        """
        for i in range(batch_data.shape[0]):
            self.add_window(batch_data[i])

    def get_result(self):
        """获取最终重构结果"""
        result = self.results / np.maximum(self.counts, 1.0)
        return result

    def reset(self):
        """重置状态，用于处理新的数据集"""
        self.results.fill(0)
        self.counts.fill(0)
        self.window_idx = 0


def model_inference(
        data_loader,
        model,
        device,
        original_length,
        seq_len,
        n_features,
        inference_fn=None,
        extract_fn=None,
        progress=False
):
    """
    通用的流式重构函数

    Args:
        data_loader: DataLoader
        model: 模型
        device: 设备
        original_length: 原始序列长度
        seq_len: 窗口长度
        n_features: 特征维度
        inference_fn: 自定义推理函数 (batch_data, model, device) -> model_output
                     如果为 None，使用默认的 model(batch_data)
        extract_fn: 从模型输出中提取重构结果的函数 (model_output) -> numpy array
                    如果为 None，假设模型输出直接是结果
        progress: 是否显示进度

    Returns:
        reconstructed: numpy array, shape [original_length, n_features]
    """
    model.eval()
    streamer = StreamingInference(original_length, seq_len, n_features)

    with torch.no_grad():
        for batch_idx, batch_item in enumerate(data_loader):
            # 1. 自定义推理逻辑
            if inference_fn is not None:
                model_output = inference_fn(batch_item, model, device)
            else:
                # 默认逻辑：假设 batch_item 是 (data, label) 格式
                if isinstance(batch_item, (list, tuple)):
                    batch_data = batch_item[0].to(device)
                else:
                    batch_data = batch_item.to(device)
                model_output = model(batch_data)

            # 2. 提取重构结果
            if extract_fn is not None:
                rec_batch_np = extract_fn(model_output)
            else:
                # 默认逻辑：假设输出直接是 tensor
                if isinstance(model_output, torch.Tensor):
                    rec_batch_np = model_output.cpu().numpy()
                else:
                    raise ValueError("extract_fn is required for non-tensor output")

            # 3. 添加到结果中
            streamer.add_batch(rec_batch_np)

            # 4. 及时释放内存
            del batch_item, model_output, rec_batch_np

            if (batch_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 5. 显示进度
            if progress and (batch_idx + 1) % 10 == 0:
                prog = (batch_idx + 1) / len(data_loader) * 100
                print(f"Progress: {prog:.1f}%", end='\r')

    if progress:
        print()  # 换行

    return streamer.get_result()


"""
def complex_inference_fn(batch_item, model, device):
    batch_data, _ = batch_item
    batch_data = batch_data.to(device)
    
    # 构造多个输入
    x_enc = batch_data
    x_mark_enc = torch.zeros(batch_data.shape[0], batch_data.shape[1], 4, device=device)
    
    # 模型可能返回多个输出
    outputs = model(x_enc, x_mark_enc, None, None)
    
    return outputs

def complex_extract_fn(model_output):
    # 假设输出是 (reconstruction, hidden, attention_weights)
    reconstruction, hidden, attn = model_output
    return reconstruction.cpu().numpy()
"""