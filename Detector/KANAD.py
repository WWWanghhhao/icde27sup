import torch as th
from torch import nn

class KANAD(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.window = configs.seq_len
        self.order = configs.order
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine().unsqueeze(0),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, kernel_size=1, padding=0, stride=1, dilation=1)

    def forward(self, x: th.Tensor, *args, **kwargs):
        # x: (batch, window, features)
        if x.dim() != 3:
            raise ValueError("Expected input shape (batch, window, features)")
        b, w, f = x.shape

        # reshape -> (batch*features, window)
        x = x.permute(0, 2, 1).contiguous().view(b * f, w)

        res = []
        res.append(x.unsqueeze(1))
        ff = th.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [th.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # (batch*features, channels, window)
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)  # (batch*features, 1, window)

        out = ff.squeeze(1)          # (batch*features, window)
        out = out.view(b, f, w)      # (batch, features, window)
        out = out.permute(0, 2, 1)   # (batch, window, features)
        return out

    def _create_custom_periodic_cosine(self) -> th.Tensor:
        pl = [i for i in range(1, self.order + 1)]
        result = th.empty(self.order, self.window, dtype=th.float32)
        for i, p in enumerate(pl):
            range_value = th.arange(self.window, dtype=th.float32)
            result[i, :] = th.cos(2 * th.pi * range_value * p / self.window)
        return result