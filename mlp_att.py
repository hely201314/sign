import math
import torch


from collections import OrderedDict
import math


import torch as th
import torch.nn as nn



from timm.models.layers import DropPath

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


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class GlossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim , bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.Lin_att = nn.Linear(dim,self.num_heads , bias=qkv_bias)
        #self.attpool = AttentionPool2d(dim,self.num_heads, self.num_heads)

        self.att_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x,cls):
        B, N, C = cls.shape

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(cls).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k v = 12,8,224,128  x = 12,224,1024 q = 1,8,224,128
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)#12,8,224,224
        #add
        # unfold = self.unfold(x,T=2)
        # l = torch.rand((12,8,224,224))
        #l = self.Lin_att(x).reshape(B, N, self.num_heads, -1)
        #l = 12,8,224,128

        x = self.Lin_att(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(B,self.num_heads,-1)
        x_mean = x.mean(-1, keepdim=True).unsqueeze(-1)  # 12,8,1
        x_max = x.max(-1, keepdim=True)[0].unsqueeze(-1)
        # #x1 = x.permute(0,2,1)
        # #l = self.attpool(x1)
        # #l = l.unsqueeze(-1).unsqueeze(-1)
        # #l = (1/3*l+1/3*x_mean+1/3*x_max)*attn
        # # l = (l)*attn
        #
        x = (0.5*x_mean+0.5*x_max)*attn
        x = x.softmax(dim=-1)



        attn = self.att_weight*x + (1-self.att_weight)*attn
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads = 12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = GlossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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

    def forward(self, x,cls):
        cls = cls + self.attn(self.norm1(x),self.norm1(cls))
        #cls = ((2 - 1) * +1 * self.drop_path(self.attn(self.norm1(x))))
        x = cls + self.mlp(self.norm2(cls))
        return x

def test():
    x = torch.rand(6, 2,224,768)#T = 2
    x = x.reshape(-1, 224,768)
    z = torch.rand(12,12,224,224)
    query = nn.Parameter(torch.rand(1, 1, 768), requires_grad=True)
    query = query.repeat(1, 224, 1)  # 1,224,768
    model = Block(768,12)
    y = model(x,query)
    print(y.shape)
    # att = AttentionPool2d(32,1024,8,8)
    # y = att(x)
    # print(y.shape)
    # y = y.reshape(12,8,1,1)
    # print(y.shape)
    # z = y*z
    # print(z.shape)
    # #
    #y = model(x)




def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        #self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)

        #x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)






#test()

