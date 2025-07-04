# -*- coding: utf-8 -*-
# @Time    : 10/22/21 1:23 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gopt.py

# attention part is borrowed from the timm package.

import math
import warnings
import torch
import torch.nn as nn
from torchaudio.models import Conformer
import numpy as np

# code from the t2t-vit paper
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #print(C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    # TODO etiher delete this class from this file or change it to make it a Conformer block
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# standard GOPT model proposed in the paper, but replace the Transformer with a Conformer
class GOPC(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        # Transformer encode blocks
        #self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads) for i in range(depth)])
        # Conformer encoder
        self.conformer_encoder = Conformer(
            input_dim=self.embed_dim, 
            num_heads=num_heads,
            num_layers=depth,
            ffn_dim=self.embed_dim,
            depthwise_conv_kernel_size=31
        )

        # sin pos embedding or learnable pos embedding, 55 = 50 sequence length + 5 utt-level cls tokens
        #self.pos_embed = nn.Parameter(get_sinusoid_encoding(55, self.embed_dim) * 0.1, requires_grad=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 55, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # for phone classification
        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # for word classification, 1=accuracy, 2=stress, 3=total
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # canonical phone projection, assume there are 40 phns
        self.phn_proj = nn.Linear(40, embed_dim)

        # utterance level, 1=accuracy, 2=completeness, 3=fluency, 4=prosodic, 5=total score
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token5 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # initialize the cls tokens
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)
        trunc_normal_(self.cls_token5, std=.02)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, phn):

        # batch size
        B = x.shape[0]

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot = torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        # if the input dimension is different from the Transformer embedding dimension, project the input to same dim
        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed

        cls_token1 = self.cls_token1.expand(B, -1, -1)
        cls_token2 = self.cls_token2.expand(B, -1, -1)
        cls_token3 = self.cls_token3.expand(B, -1, -1)
        cls_token4 = self.cls_token4.expand(B, -1, -1)
        cls_token5 = self.cls_token5.expand(B, -1, -1)

        x = torch.cat((cls_token1, cls_token2, cls_token3, cls_token4, cls_token5, x), dim=1)

        x = x + self.pos_embed

        # forward to the Conformer encoder
        seq_len = x.shape[1]
        x = self.conformer_encoder(
            input = x,
            lengths = torch.full(size=(B,), fill_value=seq_len)
        )[0] # Conformer's forward method returns (output frames, output lengths)

        # the first 5 tokens are utterance-level cls tokens, i.e., accuracy, completeness, fluency, prosodic, total scores
        u1 = self.mlp_head_utt1(x[:, 0])
        u2 = self.mlp_head_utt2(x[:, 1])
        u3 = self.mlp_head_utt3(x[:, 2])
        u4 = self.mlp_head_utt4(x[:, 3])
        u5 = self.mlp_head_utt5(x[:, 4])

        # 6th-end tokens are phone score tokens
        p = self.mlp_head_phn(x[:, 5:])

        # word score is propagated to phone-level, so word output is also at phone-level.
        # but different mlp heads are used, 1 = accuracy, 2 = stress, 3 = total
        w1 = self.mlp_head_word1(x[:, 5:])
        w2 = self.mlp_head_word2(x[:, 5:])
        w3 = self.mlp_head_word3(x[:, 5:])
        return u1, u2, u3, u4, u5, p, w1, w2, w3