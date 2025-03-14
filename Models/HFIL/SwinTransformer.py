# -*- coding: utf-8 -*-
"""
@author: gaohaoran@Dalian Minzu University
@software: PyCharm
@file: HFIL-Net.py
@time: 2023/11/23 7:23
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C) 堆叠到一起形成一个长条
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每一个头的通道维数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1---Important！！！
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # 输入此的x是整图
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('FFN',x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans  # define in_chans == 3
        self.embed_dim = embed_dim  # Swin-B.embed_dim ==128,(T is 96)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # dim 3->128
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints,尺寸固定，下有断言
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=None, num_heads=None,
                 window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            # self.layers 中应该是 4 个
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        layer_features = []
        x = self.patch_embed(x)
        B, L, C = x.shape
        layer_features.append(x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous())

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
            B, L, C = x.shape
            xl = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
            layer_features.append(xl)
        x = self.norm(x)  # B L C
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        layer_features[-1] = x

        return layer_features

    def forward(self, x):
        outs = self.forward_features(x)

        return outs

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class HFILNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(HFILNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.patch_embed = PatchEmbed(
            img_size=384, patch_size=4, in_chans=3, embed_dim=128, norm_layer=norm_layer)

        self.LFF_1 = LFF(128, 96)
        self.LFF_2 = LFF(256, 48)
        self.LFF_3 = LFF(512, 24)
        self.HFF_1 = HFF(1024, 12)
        self.HFF_2 = HFF(1024, 12)

        self.SG = SG(1024, 12)

        self.CA = ChannelAttention(1024)
        self.CC1 = CC(1024, 1024, 1024)
        self.CC2 = CC(512, 1024, 512)
        self.CC3 = CC(256, 1024, 256)
        self.SAM1 = SpatialAttention()
        self.SAM2 = SpatialAttention()
        self.CC4 = CC(1024, 512, 512)
        self.CC5 = CC(512, 256, 256)
        self.CC6 = CC(1024, 256, 256)
        self.CC7 = CC(128, 256, 128)
        self.EDGE = EDGE(512, 256, 128, 128)

        self.conv_sg = nn.Conv2d(1024, 1, 3, padding=1)
        self.conv_s = nn.Conv2d(128, 1, 3, padding=1)
        self.conv_s2 = nn.Conv2d(512, 1, 3, padding=1)
        self.conv_s3 = nn.Conv2d(256, 1, 3, padding=1)
        self.conv_se = nn.Conv2d(128, 1, 3, padding=1)

        self.conv_c3 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)


        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, mode):
        dtype = self.patch_embed.proj.weight.dtype
        bs, h, w, c = data["img"].shape
        img = data["img"].permute([0, 3, 1, 2]).to(dtype)
        depth = data["depth"].permute([0, 3, 1, 2]).to(dtype)
        seg = data["seg"].permute([0, 3, 1, 2]).to(dtype)

        tgt = data["tgt"].permute([0, 3, 1, 2]).to(dtype)
        tgt = torch.mean(tgt, dim=1, keepdim=True)
        tgt_edge = data["edge"].reshape(bs, h, w, 1).permute([0, 3, 1, 2]).to(dtype)
        tgt_edge = torch.mean(tgt_edge, dim=1, keepdim=True)

        rgb_list = self.rgb_swin(img)
        depth_list = self.depth_swin(depth)
        c = self.patch_embed(seg)

        B, L, C = c.shape
        c = c.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()

        r1 = rgb_list[0]  # (8,128,96,96)
        r2 = rgb_list[1]  # (8,256,48,48)
        r3 = rgb_list[2]  # (8,512,24,24)
        r4 = rgb_list[3]  # (8,1024,12,12)
        r5 = rgb_list[4]  # (8,1024,6,6)

        d1 = depth_list[0]  # (8,128,96,96)
        d2 = depth_list[1]  # (8,256,48,48)
        d3 = depth_list[2]  # (8,512,24,24)
        d4 = depth_list[3]  # (8,1024,12,12)
        d5 = depth_list[4]  # (8,1024,6,6)

        # c1 = self.LFF_1(r1, d1, None).repeat(1, 2, 1, 1)
        c1 = self.LFF_1(r1, d1, c)
        c2 = self.LFF_2(r2, d2, c1)
        c3 = self.LFF_3(r3, d3, c2)
        c4 = self.HFF_1(r4, d4, self.conv_c3(c3))
        c5 = self.HFF_2(r5, d5, c4)

        sg = self.SG(c4, c5)

        x = sg * self.CA(sg)

        x1 = F.interpolate(x, [24, 24])
        x2 = F.interpolate(x, [48, 48])
        x3 = F.interpolate(x, [96, 96])

        s1 = self.CC1(c3, x1)
        s2 = self.CC2(c2, x2)
        s3 = self.CC3(c1, x3)

        s1_1 = self.SAM1(s1) * s1
        s2_ = self.CC4(F.interpolate(s1_1, scale_factor=2), s2)

        s1_ = self.SAM2(s2_) * s2_
        s1_ = self.CC5(F.interpolate(s1_, scale_factor=2), s3)

        s1_ = self.CC6(F.interpolate(s1_1, scale_factor=4), s1_)

        edge = self.EDGE(d3, d2, d1)

        s = self.CC7(edge, s1_)

        s = F.interpolate(self.conv_s(s), [384, 384], mode="bilinear")
        s2 = F.interpolate(self.conv_s2(s2_), [384, 384], mode="bilinear")
        s3 = F.interpolate(self.conv_s3(s3), [384, 384], mode="bilinear")
        edge = F.interpolate(self.conv_se(edge), [384, 384], mode="bilinear")
        sg = F.interpolate(self.conv_sg(sg), [384, 384], mode="bilinear")

        loss_sg = self.loss(sg, tgt) + self.iou_loss(sg, tgt)
        loss1 = self.loss(s, tgt) + self.iou_loss(s, tgt)
        loss2 = self.loss(s2, tgt) + self.iou_loss(s2, tgt)
        loss3 = self.loss(s3, tgt) + self.iou_loss(s3, tgt)
        loss_edge = self.loss(edge, tgt_edge)

        return loss_sg + loss1 + loss2 + loss3 + loss_edge, self.sigmoid(s), tgt

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

    def iou_loss(self, pred, mask):
        pred = torch.sigmoid(pred)
        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        return iou.mean()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class SG(nn.Module):
    def __init__(self, in_channels, shape):
        super(SG, self).__init__()
        self.AAP = nn.AdaptiveAvgPool2d(shape)
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1)

        self.a_conv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, dilation=1, padding=2)
        self.a_conv3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, dilation=1, padding=2)
        self.a_conv5_1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, dilation=1, padding=2)
        self.a_conv5_2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, dilation=1, padding=2)

        self.CBR = CBR(in_channels)

    def forward(self, c4, c5):
        x5 = torch.sigmoid(self.conv1_1(self.AAP(c4)))

        c5_1 = self.conv1_2(c5)
        c5_2 = self.a_conv3_1(c5)
        c5_3 = self.a_conv5_1(c5)
        # print(c5_1.shape, c5_2.shape, c5_3.shape)
        c5 = torch.cat([c5_1, c5_2, c5_3], dim=1)
        c5 = self.conv3_1(c5)

        c4_1 = self.conv1_3(c4)
        c4_2 = self.a_conv3_2(c4)
        c4_3 = self.a_conv5_2(c4)
        c4 = torch.cat([c4_1, c4_2, c4_3], dim=1)
        c4 = self.conv3_2(c4)

        x = torch.sigmoid(c4) * c5 + c5 * x5

        return self.CBR(x)


class EDGE(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3, out_channels):
        super(EDGE, self).__init__()
        self.CA = ChannelAttention(out_channels*3)
        self.CBR = CBR(out_channels*3)
        self.conv = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        self.conv1_1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels3, out_channels, kernel_size=1)
        self.up_2 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=4)

    def forward(self, x1, x2, x3):
        x1 = self.up_4(x1)
        x2 = self.up_2(x2)
        x1 = self.conv1_1(x1)
        x2 = self.conv1_2(x2)
        x3 = self.conv1_3(x3)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.CA(self.CBR(x)) * x + x
        return self.conv(x)

class CC(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super(CC, self).__init__()
        self.conv = nn.Conv2d(in_channels_1+in_channels_2, out_channels, 1)


    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


# Cross-Modal Cross-Scale Interaction Block
class LFF(nn.Module):
    def __init__(self, in_channels, shape):
        super(LFF, self).__init__()
        self.shape = shape
        self.SA_rgb = SpatialAttention()
        self.AAP_rgb = nn.AdaptiveAvgPool2d(output_size=shape)
        self.SA_d = SpatialAttention()
        self.AAP_d = nn.AdaptiveAvgPool2d(output_size=shape)
        self.CC_r = CC(in_channels, in_channels, in_channels)
        self.CC_d = CC(in_channels, in_channels, in_channels)
        self.CC_1 = CC(in_channels, in_channels, in_channels)
        self.CC_2 = CC(in_channels, in_channels, in_channels)

        self.softmax = nn.Softmax()


    def forward(self, x_rgb, x_d, c=None):
        x_rgb_raw = x_rgb
        x_rgb_1 = x_rgb * self.SA_rgb(x_rgb)
        x_rgb_2 = x_rgb * self.softmax(self.AAP_rgb(x_rgb))
        x_rgb_2 = self.CC_r(x_rgb_raw, x_rgb_2)
        x_rgb_3 = x_rgb_1 * x_rgb_2
        x_rgb = x_rgb_1 + x_rgb_2 + x_rgb_3

        x_d = x_d * x_rgb
        x_d_1 = x_d * self.SA_d(x_d)
        x_d_2 = x_d * self.softmax(self.AAP_d(x_d))
        x_d_2 = self.CC_d(x_d_2, x_d)
        x_d_3 = x_d_1 * x_d_2
        x_d = x_d_1 + x_d_2 + x_d_3
        x_rgb = x_rgb_raw * x_d

        x = self.CC_1(x_rgb, x_d)
        if c is not None:
            c = F.interpolate(c, self.shape)
            x = self.CC_2(x, c)
        return x


class HFF(nn.Module):
    def __init__(self, in_channels, shape):
        super(HFF, self).__init__()
        self.CA_r = ChannelAttention(in_channels)
        self.CA_c = ChannelAttention(in_channels)
        self.CA_d = ChannelAttention(in_channels)

        self.CBR_r_1 = CBR(in_channels)
        self.CBR_r_2 = CBR(in_channels)
        self.CBR_c_1 = CBR(in_channels)
        self.CBR_c_2 = CBR(in_channels)
        self.CBR_d_1 = CBR(in_channels)
        self.CBR_d_2 = CBR(in_channels)

        self.AMP = nn.AdaptiveMaxPool2d(shape)
        self.AAP = nn.AdaptiveAvgPool2d(shape)
        self.sigmoid = nn.Sigmoid()
        self.CC = CC(in_channels, in_channels, in_channels)

    def forward(self, x_r, x_d, x_c):
        x_r = self.CA_r(x_r) * x_r
        x_d = self.CA_d(x_d) * x_d
        x_c = self.CA_c(x_c) * x_c

        x_r_ = self.CBR_r_1(self.CBR_r_2(x_r * x_c))
        x_c_ = self.sigmoid(self.CBR_c_1(self.AAP(x_r * x_d)) + self.CBR_c_1(self.AMP(x_r * x_d)))
        x_d_ = self.CBR_d_1(self.CBR_d_2(x_d * x_c))

        x_r = x_r_ * x_c_
        x_d = x_d_ * (1 - x_c_)

        x_r_ = x_r + x_d
        x_d_ = x_r * x_d

        return self.CC(x_r_, x_d_)


class CBR(nn.Module):
    def __init__(self, in_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2 * math.pi)
        self.initialize()

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        a = a_in.view(b, l, -1, 1, 1)

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_ * (self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq * self.ln_2pi), dim=-1) \
                 - torch.sum((v - mu) ** 2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l ** (1 / 2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out

