import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import einops
import torch.nn.functional as F



class ScaledDotProductAttention(nn.Module):
    def __init__(self,dim,head):
        super(ScaledDotProductAttention, self).__init__()
        head_dim = dim // head
        self.scale = head_dim ** -0.5
        self.predictor1 = nn.Sequential(
            nn.Linear(head_dim,head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim,head_dim),
            nn.Tanh(),
        )
        self.predictor2 = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim),
            nn.Tanh(),
        )


    def forward(self,Q, K, V, attn_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """

        scores = torch.matmul(Q, K.transpose(-1, -2))*self.scale # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        #scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        temq = self.predictor1(Q)#(b,head,L,headdim)
        temq = torch.mean(temq,dim=-1).unsqueeze(-1)
        temq = nn.Softmax(dim=-1)(temq)



        attn = (temq-0.5)*attn


        context = torch.matmul(attn, V).transpose(1, 2)# context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        #context = 1,224,12,64

        return context, attn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Block(nn.Module):

    def __init__(self, dim, num_heads = 12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim)
        self.attn = ScaledDotProductAttention(dim,num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 1)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 1, dim))
        ]))
        self.temporal_ada_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.W_Q = nn.Linear(dim, dim,bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(dim, dim,bias=False)
        self.W_V = nn.Linear(dim, dim,bias=False)



    def forward(self, x,cls):
        x = self.norm1(torch.cat((cls, x),0))
        q,k,v = x[:1], x[1:], x[1:]
        residual,B = q,q.shape[0]
        q = self.W_Q(q).view(B,-1,self.num_heads,self.dim //self.num_heads).transpose(1, 2)#1,12,224,64
        k = self.W_K(k).view(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)#1,12,2688,64
        v = self.W_V(v).view(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)#1,12,2688,64
        q= self.attn(q, k, v)[0].contiguous() .view(B, -1, self.dim)
        q = residual + q
        #cls = ((2 - 1) * +1 * self.drop_path(self.attn(self.norm1(x))))
        q = q + self.mlp(self.norm2(q))
        return q

class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)



    def forward(self, x,cls):
        B,L,C = cls.shape
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out
class relu_linear_att(nn.Module):
    def __init__(self, dim=512, token_dim=256, num_heads=12):
        super().__init__()

        self.dim = dim
        # self.W_Q = nn.Linear(dim, dim, bias=False)  # q,k必须维度相同，不然无法做点积
        # self.W_K = nn.Linear(dim, dim, bias=False)
        # self.W_V = nn.Linear(dim, dim, bias=False)
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim)
        self.kernel_func = nn.ReLU()
        #self.eps = 1.0e-15,
    def forward(self,q, k, v) -> torch.Tensor:
        # x = self.norm1(torch.cat((cls, x), 0))
        # q, k, v = x[1:], x[1:], x[1:]
        residual, B = q, q.shape[0]
        # q = self.W_Q(q).view(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)#()
        # k = self.W_K(k).view(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        # v = self.W_V(v).view(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        #cls = (1,224,768) kv=1,12,12*224,64 q = 1,12,224,64

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)#v = 1,12,224*12+1,64
        vk = torch.matmul(v, trans_k)#1,12,2689,2688
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        #out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        out = out[:, :, :-1] / (out[:, :, -1:])
        out = out.transpose(1, 2).reshape(B,-1,self.dim)



        return out



def test():
    x = torch.rand(6, 2,224,768)#T = 2
    x = x.reshape(-1, 224,768)

    query = nn.Parameter(torch.rand(1, 1, 768), requires_grad=True)
    query = query.repeat(1, 224, 1)  # 1,224,768


    model = Block(768,12)
    y = model(x,query)
    print(y.shape)

    # effm = EfficientAdditiveAttnetion(768, 768, 12)
    # y = effm(x,query)
    # print(y.shape)
    #x = torch.rand(6, 768, 224, 224)  # T = 2
    # y = relu_linear_att(768)(x,query)
    # print(y.shape)


# test()