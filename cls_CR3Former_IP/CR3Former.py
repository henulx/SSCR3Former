import PIL
import time
import torch
import math
import torchvision
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from thop import profile
from scipy.io import savemat
import numpy as np


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv2Former_Mod(nn.Module):
    def __init__(self, dim, K):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, K, padding= K//2, groups=dim),
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class MCM(nn.Module):
    def __init__(self, dim, K, drop_path=0.):
        super().__init__()

        self.attn = Conv2Former_Mod(dim, K)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))

        return x

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3 size 64 192
        self.dim = dim
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def differential_attention(self, scores):##差分计算
        diff_scores = scores[:, 1:] - scores[:, :-1]
        pld = (0,0,0,0,1,0)#torch.Size([64, 8, 5, 5])
        diff_scores = torch.nn.functional.pad(diff_scores, pld, 'constant',0)
        return diff_scores

    def forward(self, x,mask=None):
        b, n, _, h = *x.shape, self.heads#x.shape:[64, 5 ,64] [b=128 n=13 _=64]
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots1 = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  ##torch.Size([64, 8, 5, 5])
        dots2 = self.differential_attention(dots1) * self.scale
        dots = dots1 - dots2
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        S = dots[:, 0]  ##torch.Size([64, 5, 5])
        # S = torch.tanh_(S) ##torch.Size([64, 5, 5])
        # m = nn.Mish()
        # S = m(S)
        m = nn.Softplus()
        S = m(S)
        # k = nn.Softshrink()
        # S = k(S)
        # m = nn.Tanhshrink()
        # S = m(S)
        # S[...,0] = 0
        # S = (1-torch.eye(n))*S
        S = torch.roll(S, 1, -2)
        # S[..., 0, :] = 0
        F1 = torch.cumsum(S, dim=-2)
        dots = dots - F1[:, None]
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper [128 8 13 13]
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax 128 8 13 8
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out      #torch.Size([128, 13, 64])


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,pj_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                # Residual(LayerNormalize(dim, AttentionTSSA(dim, num_heads=heads, qkv_bias=False,attn_drop=dropout,
                #                                            proj_drop=pj_dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

NUM_CLASS = 16
# NUM_CLASS = 15
# NUM_CLASS = 9
# NUM_CLASS = 20
# NUM_CLASS = 22
# NUM_CLASS = 6

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1,num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1,
                 pj_dropout=0):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.scale = dim ** -0.5
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3d_features5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(5, 5, 5)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.norm = nn.BatchNorm2d(32)
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim,dropout,pj_dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.mcm = MCM(224,11)

    def forward(self, x, mask=None):
        y = x
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        y = rearrange(y, 'b c h w y -> b (c h) w y')
        y = self.conv2d_features1(y)
        x = self.mcm(x)
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')
        y = rearrange(y,'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        B = torch.einsum('bij,bjk->bik', y, wa)
        B = rearrange(B, 'b h w -> b w h')  # Transpose
        B = B.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        VVb = torch.einsum('bij,bjk->bik', y, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        Tb = torch.einsum('bij,bjk->bik', B, VVb)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_tokenbs = self.cls_token.expand(y.shape[0], -1, -1)
        y = torch.cat((cls_tokenbs, Tb), dim=1)
        y += self.pos_embedding
        y = self.dropout(y)  # torch.Size([128, 13, 64])
        y = self.transformer(y)  # size:[128 13 64]
        y = self.to_cls_token(y[:, 0])
        y = self.nn1(y)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)   #torch.Size([128, 13, 64])
        x = self.transformer(x)  # size:[128 13 64]
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)
        z = y + x
        return z

def cal_time1(model, x):
    with torch.inference_mode():
        time_list = []
        for _ in range(50):
            ts = time.perf_counter()
            ret = model(x)
            td = time.perf_counter()
            time_list.append(td - ts)

        print(f"avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f}")

if __name__ == '__main__':
    model = SSFTTnet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 11, 11)
    cal_time1(model, input)
    flops, params = profile(model, (input,))
    print('flops: ', str(flops / 1024 ** 3) + 'G', 'params: ', str(params / 1024 ** 2) + 'M')
    y = model(input)
    print(y.size())


