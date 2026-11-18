# timm portions: Copyright Ross Wightman (Apache-2.0); RVT portions: MIT.
# Modified for FE-FusionPoints; see THIRD_PARTY_LICENSES.txt.
"""
Part of this code stems from rwightman's MaxVit implementation:
https://github.com/huggingface/pytorch-image-models/blob/1885bdc4318cc3be459981ea1a26cd862220864d/timm/models/maxxvit.py
that is:
- LayerScale
- PartitionAttentionCl
- window*
- grid*
- SelfAttentionCl
"""

from enum import Enum, auto
from functools import partial
from typing import Optional, Union, Tuple, List, Type

import math
import torch
from torch import nn

from .layers import DropPath, LayerNorm
from .layers import get_act_layer, get_norm_layer
from .layers import to_2tuple, _assert


class PartitionType(Enum):
    WINDOW = auto()
    GRID = auto()


def nChw_2_nhwC(x: torch.Tensor):
    """N C H W -> N H W C
    """
    assert x.ndim == 4
    return x.permute(0, 2, 3, 1)


def nhwC_2_nChw(x: torch.Tensor):
    """N H W C -> N C H W
    """
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class GLU(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 channel_last: bool,
                 act_layer: Type[nn.Module],
                 bias: bool = True):
        super().__init__()
        # Different activation functions / versions of the gated linear unit:
        # - ReGLU:  Relu
        # - SwiGLU: Swish/SiLU
        # - GeGLU:  GELU
        # - GLU:    Sigmoid
        # seem to be the most promising once.
        # Extensive quantitative eval in table 1: https://arxiv.org/abs/2102.11972
        # Section 2 for explanation and implementation details: https://arxiv.org/abs/2002.05202
        # NOTE: Pytorch has a native GLU implementation: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html?highlight=glu#torch.nn.GLU
        proj_out_dim = dim_out*2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)
        self.channel_dim = -1 if channel_last else 1

        self.act_layer = act_layer()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)


class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 channel_last: bool,
                 expansion_ratio: int,
                 act_layer: Type[nn.Module],
                 gated: bool = True,
                 bias: bool = True,
                 drop_prob: float = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            # To keep the number of parameters (approx) constant regardless of whether glu == True
            # Section 2 for explanation: https://arxiv.org/abs/2002.05202
            #inner_dim = round(inner_dim * 2 / 3)
            #inner_dim = math.ceil(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            #inner_dim = round(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, act_layer=act_layer, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias) if channel_last else \
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                act_layer(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class DownsampleBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def output_is_normed():
        raise NotImplementedError


def get_downsample_layer_Cf2Cl(dim_in: int,
                               dim_out: int,
                               downsample_factor: int,
                               downsample_cfg: dict) -> DownsampleBase:
    type = downsample_cfg["type"]
    if type == 'patch':
        return ConvDownsampling_Cf2Cl(dim_in=dim_in,
                                      dim_out=dim_out,
                                      downsample_factor=downsample_factor,
                                      downsample_cfg=downsample_cfg)
    raise NotImplementedError


class ConvDownsampling_Cf2Cl(DownsampleBase):
    """Downsample with input in NCHW [channel-first] format.
    Output in NHWC [channel-last] format.
    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 downsample_factor: int,
                 downsample_cfg: dict):
        super().__init__()
        assert isinstance(dim_out, int)
        assert isinstance(dim_in, int)
        assert downsample_factor in (2, 4, 8)

        norm_affine = downsample_cfg["norm_affine"]
        overlap = downsample_cfg["overlap"]

        if overlap:
            kernel_size = (downsample_factor - 1)*2 + 1
            padding = kernel_size//2
        else:
            kernel_size = downsample_factor
            padding = 0
        self.conv = nn.Conv2d(in_channels=dim_in,
                              out_channels=dim_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=downsample_factor,
                              bias=False)
        self.norm = LayerNorm(num_channels=dim_out, eps=1e-5, affine=norm_affine)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = nChw_2_nhwC(x)
        x = self.norm(x)
        return x

    @staticmethod
    def output_is_normed():
        return True


class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.

    According to RW, NHWC attention is a few percent faster on GPUs (but slower on TPUs)
    https://github.com/rwightman/pytorch-image-models/blob/4f72bae43be26d9764a08d83b88f8bd4ec3dbe43/timm/models/maxxvit.py#L1258
    """

    def __init__(
            self,
            dim: int,
            partition_type: PartitionType,
            attention_cfg: dict,
            skip_first_norm: bool=False,
    ):
        super().__init__()
        norm_eps = attention_cfg["norm_eps"]
        partition_size = attention_cfg["partition_size"]
        use_torch_mha = attention_cfg["use_torch_mha"]
        dim_head = attention_cfg["dim_head"]
        attention_bias = attention_cfg["attention_bias"]
        mlp_act_string = attention_cfg["mlp_activation"]
        mlp_gated = attention_cfg["mlp_gated"]
        mlp_bias = attention_cfg["mlp_bias"]
        mlp_expand_ratio = attention_cfg["mlp_ratio"]

        drop_path = attention_cfg["drop_path"]
        drop_mlp = attention_cfg["drop_mlp"]
        ls_init_value = attention_cfg["ls_init_value"]

        assert isinstance(use_torch_mha, bool)
        assert isinstance(mlp_gated, bool)
        assert_activation_string(activation_string=mlp_act_string)
        mlp_act_layer = get_act_layer(mlp_act_string)

        self_attn_module = TorchMHSAWrapperCl if use_torch_mha else SelfAttentionCl

        if isinstance(partition_size, int):
            partition_size = to_2tuple(partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        norm_layer = partial(get_norm_layer('layernorm'), eps=norm_eps)  # NOTE this block is channels-last

        assert isinstance(partition_type, PartitionType)
        self.partition_window = partition_type == PartitionType.WINDOW

        self.norm1 = nn.Identity() if skip_first_norm else norm_layer(dim)
        self.self_attn = self_attn_module(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias)
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim = dim,
                       channel_last=True,
                       expansion_ratio = mlp_expand_ratio,
                       act_layer = mlp_act_layer,
                       gated = mlp_gated,
                       bias = mlp_bias,
                       drop_prob = drop_mlp)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_window:
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.self_attn(partitioned)

        if self.partition_window:
            x = window_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        else:
            x = grid_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def window_partition(x, window_size: Tuple[int, int]):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: Tuple[int, int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, f'width {W} must be divisible by grid {grid_size[1]}')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class TorchMHSAWrapperCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        assert dim % dim_head == 0
        num_heads = dim // dim_head
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor):
        restore_shape = x.shape
        B, C = restore_shape[0], restore_shape[-1]
        x = x.view(B, -1, C)
        attn_output, attn_output_weights =  self.mha(query=x, key=x, value=x)
        attn_output = attn_output.reshape(restore_shape)
        return attn_output


class SelfAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x

class CrossAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.q_layer = nn.Linear(dim, dim, bias=bias)
        self.k_layer = nn.Linear(dim, dim, bias=bias)
        self.v_layer = nn.Linear(dim, dim, bias=bias)

        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor):
        B = x_0.shape[0]
        restore_shape = x_0.shape[:-1]

        q = self.q_layer(x_0).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.k_layer(x_1).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v_layer(x_1).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x
    
class CrossChannelAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            bias: bool = True):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.q_layer = nn.Linear(dim, dim, bias=bias)
        self.k_layer = nn.Linear(dim, dim, bias=bias)
        self.v_layer = nn.Linear(dim, dim, bias=bias)

        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor):
        B = x_0.shape[0]
        restore_shape = x_0.shape[:-1]

        q = self.q_layer(x_0).view(B, -1, self.dim)
        k = self.k_layer(x_1).view(B, -1, self.dim)
        v = self.v_layer(x_1).view(B, -1, self.dim)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x
    

class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        config,
        need_norm = True):
        super().__init__()

        self.dim = in_channels
        self.dim_head = config["attention"]["dim_head"]
        self.attention_bias = config["attention"]["attention_bias"]

        if need_norm:
            self.norm0 = nn.LayerNorm(self.dim)
            self.norm1 = nn.LayerNorm(self.dim)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()

        self.attn = CrossAttentionCl(self.dim, self.dim_head, self.attention_bias)

        self.norm2 = nn.LayerNorm(self.dim)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=2*self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.dim,self.dim)
        )

    def forward(self, x_0:torch.Tensor, x_1:torch.Tensor):
        x = x_0 + self.attn(self.norm0(x_0),self.norm1(x_1))
        x = x + self.MLP(self.norm2(x))

        return x

class CrossChannelAttentionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        config,
        need_norm = False):
        super().__init__()

        self.dim = in_channels
        self.dim_head = config["attention"]["dim_head"]
        self.attention_bias = config["attention"]["attention_bias"]

        if need_norm:
            self.norm0 = nn.LayerNorm(self.dim)
            self.norm1 = nn.LayerNorm(self.dim)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()

        self.attn = CrossChannelAttentionCl(self.dim, self.attention_bias)

        self.norm2 = nn.LayerNorm(self.dim)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=2*self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.dim,self.dim)
        )
        if config.get("fusion_zero_init", False):
            self._init_weights()

    def _init_weights(self):
        # 1. 对 MLP 的最后一层全连接层做 0 初始化
        # 假设 MLP 最后一层是 self.MLP[2]
        nn.init.constant_(self.MLP[2].weight, 0)
        nn.init.constant_(self.MLP[2].bias, 0)

        # 2. 对 Attention 的输出层做 0 初始化
        if hasattr(self.attn, 'proj'):
             # 如果是 nn.Linear
            if isinstance(self.attn.proj, nn.Linear):
                nn.init.constant_(self.attn.proj.weight, 0)
                nn.init.constant_(self.attn.proj.bias, 0)
            # 如果是 Sequential (例如 Linear + Dropout)
            elif isinstance(self.attn.proj, nn.Sequential):
                for m in self.attn.proj:
                    if isinstance(m, nn.Linear):
                        nn.init.constant_(m.weight, 0)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x_0:torch.Tensor, x_1:torch.Tensor):
        x = x_0 + self.attn(self.norm0(x_0),self.norm1(x_1))
        x = x + self.MLP(self.norm2(x))

        return x

def assert_activation_string(activation_string: Optional[Union[str, Tuple[str, ...], List[str]]]) -> None:
    # Serves as a hacky documentation and sanity check.
    # List of possible activation layer strings that are reasonable:
    # https://github.com/rwightman/pytorch-image-models/blob/a520da9b495422bc773fb5dfe10819acb8bd7c5c/timm/models/layers/create_act.py#L62
    if activation_string is None:
        return
    if isinstance(activation_string, str):
        assert activation_string in ('silu', 'swish', 'mish', 'relu', 'relu6', 'leaky_relu', 'elu', 'prelu', 'celu', 'selu',
                             'gelu', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_swish', 'hard_mish')
    elif isinstance(activation_string, (tuple, list)):
        for entry in activation_string:
            assert_activation_string(activation_string=entry)
    else:
        raise NotImplementedError


def assert_norm2d_layer_string(norm_layer: Optional[Union[str, Tuple[str, ...], List[str]]]) -> None:
    # Serves as a hacky documentation and sanity check.
    # List of possible norm layer strings that are reasonable:
    # https://github.com/rwightman/pytorch-image-models/blob/4f72bae43be26d9764a08d83b88f8bd4ec3dbe43/timm/models/layers/create_norm.py#L14
    if norm_layer is None:
        return
    if isinstance(norm_layer, str):
        assert norm_layer in ('batchnorm', 'batchnorm2d', 'groupnorm', 'layernorm2d')
    elif isinstance(norm_layer, (tuple, list)):
        for entry in norm_layer:
            assert_norm2d_layer_string(norm_layer=entry)
    else:
        raise NotImplementedError
    
class CrossPartirionAttentionCl(PartitionAttentionCl):
    """ Grid or Block partition + Cross Attn + FFN.
    NxC 'channels last' tensor layout.
    """

    def __init__(self, dim, partition_type, attention_cfg, skip_first_norm = False):
        super().__init__(dim, partition_type, attention_cfg, skip_first_norm)

        dim_head = attention_cfg["dim_head"]
        attention_bias = attention_cfg["attention_bias"]
        norm_eps = attention_cfg["norm_eps"]

        self.self_attn = CrossAttentionCl(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias)
        
        norm_layer = partial(get_norm_layer('layernorm'), eps=norm_eps)  # NOTE this block is channels-last

        self.norm1_0 = nn.Identity() if skip_first_norm else norm_layer(dim)
        self.norm1_1 = nn.Identity() if skip_first_norm else norm_layer(dim)

    def _partition_attn(self, x_0, x_1):
        img_size = x_0.shape[1:3]
        if self.partition_window:
            partitioned_0 = window_partition(x_0, self.partition_size)
            partitioned_1 = window_partition(x_1, self.partition_size)
        else:
            partitioned_0 = grid_partition(x_0, self.partition_size)
            partitioned_1 = grid_partition(x_1, self.partition_size)

        partitioned = self.self_attn(partitioned_0, partitioned_1)

        if self.partition_window:
            x = window_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        else:
            x = grid_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        return x
    
    def forward(self, x_0, x_1):
        x = x_0 + self.drop_path1(self.ls1(self._partition_attn(self.norm1_0(x_0), self.norm1_1(x_1))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x