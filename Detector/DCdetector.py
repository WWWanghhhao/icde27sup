import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .DCdetector_layers.attn import DAC_structure, AttentionLayer
from .DCdetector_layers.embed import DataEmbedding, TokenEmbedding
from .DCdetector_layers.RevIN import RevIN
from tkinter import _flatten


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetector(nn.Module):
    def __init__(self, configs):
        """
        win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True
        """
        super(DCdetector, self).__init__()
        self.output_attention = True
        self.patch_size = [3]
        channel = configs.enc_in
        self.win_size = configs.seq_len
        enc_in = configs.enc_in
        c_out = configs.c_out
        n_heads = configs.n_heads
        d_model = configs.d_model
        dropout = configs.dropout
        self.e_layers = configs.e_layers

        # Patching List
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(self.win_size, self.patch_size, channel, False, attention_dropout=dropout,
                                  output_attention=self.output_attention),
                    d_model, self.patch_size, channel, n_heads, self.win_size) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        B, L, M = x.shape  # Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)

        # Mutil-scale Patching Operation
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size

            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))

        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None
