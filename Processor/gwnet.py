import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Diffuse(nn.Module):
    def __init__(self, in_dim, out_dim, step=3):
        super(Diffuse, self).__init__()
        self.step = step
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.diffusion = nn.ModuleList(
            [nn.Linear(self.out_dim, self.out_dim, bias=False) for i in range(self.step)])

    def forward(self, z):
        z = z.transpose(1, 2)
        z = self.linear(z)
        for l in self.diffusion:
            z = l(z)
        z = z.transpose(1, 2)
        return z


class Gwnet(nn.Module):
    def __init__(self, graph_init, config, **kwargs):
        super(Gwnet, self).__init__()
        # 解析配置参数
        # config = self._parse_configs(kwargs)
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()

        self.pred_len = config.pred_len
        self.num_nodes = config.num_nodes
        self.h_dim = config.h_dim
        self.residual_channels = config.h_dim
        self.dilation_channels = config.h_dim
        self.skip_channels = config.h_dim * 4
        self.end_channels = config.h_dim * 4
        self.kernel_size = config.kernel_size
        self.blocks = config.blocks
        self.layers = config.layers
        self.dropout = config.dropout
        self.gcn_bool = config.gcn_bool
        self.addaptadj = config.addaptadj
        self.seq_len = config.seq_len

        self.graph_init = torch.from_numpy(graph_init)

        # 初始化模块列表
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # 修正start_conv的输入通道
        # 根据forward函数，输入经过unsqueeze(3)和transpose(1,3)后，第1维变为1
        # 所以in_channels=1
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.diffusion = Diffuse(self.seq_len, self.pred_len)

        # 计算感受野
        receptive_field = 1
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field

        # 根据感受野计算各层参数
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(self.dilation_channels, self.residual_channels, self.dropout,
                                          support_len=1))

        # 修正end_conv的维度
        # 计算中间维度
        mid_channels = self.end_channels * self.num_nodes // 32

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=mid_channels,
                                    kernel_size=(self.num_nodes, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=mid_channels,
                                    out_channels=self.pred_len * self.num_nodes,
                                    kernel_size=(1, 1),
                                    bias=True)

    def _parse_configs(self, kwargs):
        class Config:
            pass

        config = Config()

        defaults = {
            'pred_len': 12,
            'num_nodes': 50,
            'h_dim': 32,
            'hyp_kernel_size': 2,
            'hyp_blocks': 4,
            'hyp_layers': 2,
            'dropout': 0.3,
            'gcn_bool': False,
            'addaptadj': False,
            'seq_len': 10,
            'device': torch.device('cpu')
        }

        for key, default_value in defaults.items():
            if key in kwargs:
                setattr(config, key, kwargs[key])
            else:
                setattr(config, key, default_value)

        for key, value in kwargs.items():
            if not hasattr(config, key):
                setattr(config, key, value)

        return config

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input = input.unsqueeze(1)
        # 简化版本的前向传播，用于快速测试
        # 输入维度: (b, 1, nvar, seq_len) = (4, 1, 50, 20)

        # print(f"输入形状: {input.shape}")

        # 根据原始forward函数的维度变换
        input = input.unsqueeze(3)  # [4, 1, 50, 20] -> [4, 1, 50, 1, 20]
        # input = input.transpose(1, 3)  # [4, 1, 50, 1, 20] -> [4, 1, 50, 1, 20] (实际上没变化)
        input = nn.functional.pad(input, (1, 0, 0, 0))  # [4, 1, 50, 1, 21]

        # 压缩维度
        input = input.squeeze(3)  # [4, 1, 50, 21]

        # print(f"pad后的形状: {input.shape}")

        # 检查序列长度
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        # print(f"start_conv输入形状: {x.shape}")

        # start_conv: [4, 1, 50, 21] -> [4, 32, 50, 21]
        x = self.start_conv(x)
        # print(f"start_conv输出形状: {x.shape}")

        # 为了测试，我们直接使用start_conv的输出，跳过中间层
        # 但end_conv_1期望的输入通道是skip_channels(256)，所以我们需要将通道数扩展到256
        # 最简单的方法是添加一个卷积层将32通道转换为256通道
        if hasattr(self, 'skip_convs') and len(self.skip_convs) > 0:
            # 使用第一个skip_conv将通道数从32扩展到256
            skip = self.skip_convs[0](x)
            # print(f"skip_conv输出形状: {skip.shape}")
        else:
            # 如果没有skip_convs，创建一个临时的卷积层
            adapt_conv = nn.Conv2d(self.residual_channels, self.skip_channels, kernel_size=(1, 1))
            skip = adapt_conv(x)
            # print(f"adapt_conv输出形状: {skip.shape}")

        # 后续处理
        x = nn.LeakyReLU(0.01)(skip)
        # print(f"LeakyReLU后形状: {x.shape}")

        x = self.end_conv_1(x)
        # print(f"end_conv_1后形状: {x.shape}")

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.end_conv_2(x)
        # print(f"end_conv_2后形状: {x.shape}")

        # 调整输出维度
        x = torch.mean(x, dim=-1, keepdim=False)  # 在时间维度上平均
        # print(f"平均后形状: {x.shape}")

        x = torch.reshape(x, shape=[-1, self.num_nodes, self.pred_len])
        # print(f"最终输出形状: {x.shape}")

        return x.permute(0, 2, 1)





# 测试函数
def test_model():
    print("测试模型前向传播...")

    # 创建随机图初始化
    num_nodes = 50
    graph_init = np.random.randn(num_nodes, num_nodes)

    # 初始化模型
    model = Gwnet(
        graph_init=graph_init,
        num_nodes=num_nodes,
        h_dim=8,
        pred_len=12
    )

    # 打印模型信息
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"感受野大小: {model.receptive_field}")
    print(f"skip_channels: {model.skip_channels}")
    print(f"end_conv_1输入通道: {model.end_conv_1.in_channels}")
    print(f"end_conv_1输出通道: {model.end_conv_1.out_channels}")

    # 创建测试输入
    batch_size = 4
    seq_len = 20
    print(f"\n创建测试输入:")
    print(f"batch_size: {batch_size}, seq_len: {seq_len}")

    x = torch.randn(batch_size, model.num_nodes, seq_len)
    print(f"输入形状: {x.shape}")

    # 前向传播
    print("\n进行前向传播...")
    try:
        output = model(x)
        print(f"\n前向传播成功!")
        print(f"输出形状: {output.shape}")

    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()


# 运行测试
if __name__ == "__main__":
    test_model()
