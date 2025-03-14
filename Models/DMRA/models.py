import cv2
import numpy as np

import xformers
import os
from xformers.ops import RMSNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import checkpoint
from Models.DMRA.backbone import RGBNet, DepthNet


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid.shape[1], grid.shape[2]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.conv1_ = nn.Conv2d(in_channels, emb_dim // 4, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.patch_embed = nn.Conv2d(in_channels=emb_dim // 4,
                                     out_channels=emb_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     padding=0)
        self.norm = norm_layer(emb_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.act(self.conv1_(x))
        x = self.patch_embed(x)
        bs, c, h, w = x.shape
        pose_embed = torch.from_numpy(get_2d_sincos_pos_embed(c, [h, w])).to(x)[None, ...]
        x = x.reshape(bs, c, -1).transpose(1, 2)
        # x = x + pose_embed
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, act_layer=nn.GELU):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels,
                             out_features=hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_channels,
                             out_features=in_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, dim, num_heads, norm=nn.LayerNorm, xformer=False):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = norm(dim//num_heads)
        self.norm_k = norm(dim//num_heads)
        self.num_heads = num_heads
        self.xformer = xformer

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        bs, n, c = q.shape
        num_heads = self.num_heads

        if self.xformer:
            q = q.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2)\
                .reshape(bs * num_heads, n, c // num_heads).contiguous()

            k = k.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2)\
                .reshape(bs * num_heads, n, c // num_heads).contiguous()

            v = v.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2)\
                .reshape(bs * num_heads, n, c // num_heads).contiguous()

            q = self.norm_q(q).to(v)
            k = self.norm_k(k).to(v)
            attn = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

            attn = attn.reshape(bs, num_heads, n, c//num_heads).transpose(1, 2).reshape(bs, n, c)

        else:
            q = q.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).contiguous()
            k = k.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).contiguous()
            v = v.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).contiguous()
            q = self.norm_q(q).to(v)
            k = self.norm_k(k).to(v)

            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)

            attn = attn.transpose(1, 2).reshape(bs, n, c)
        return self.proj(attn)


class MM_DiT_Block(nn.Module):
    def __init__(self, c_dim, num_heads, use_checkpoint=False, use_xformer=False):
        super(MM_DiT_Block, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        self.layer_norm_1 = nn.LayerNorm(c_dim)
        self.layer_norm_2 = nn.LayerNorm(c_dim)
        self.attn = AttnBlock(c_dim, num_heads, nn.LayerNorm, xformer=use_xformer)
        self.mlp = MLP(c_dim, c_dim * 4)

    def forward(self, x1, x2, x3):
        return checkpoint(
            self._forward, (x1, x2, x3), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x1, x2, x3):
        x1_len = x1.shape[1]
        x2_len = x2.shape[1]

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.attn(self.layer_norm_1(x)) + x
        x = self.mlp(self.layer_norm_2(x)) + x

        x1 = x[:, :x1_len, :]
        x2 = x[:, x1_len:x1_len+x2_len, :]
        x3 = x[:, x1_len+x2_len:, :]
        return x1, x2, x3


class FirstLayer(nn.Module):
    def __init__(self, in_channels, model_channels, patch_size=2, use_checkpoint=False):
        super().__init__()
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.model_channels = model_channels
        self.patch_emb = PatchEmbed(in_channels, model_channels, patch_size=patch_size)

    def forward(self, x):
        return checkpoint(
            self._forward, (x, ), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        x = self.patch_emb(x)
        return x

class Conv3x3Act(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.c(x))


class Conv1x1Act(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.c(x))

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out) * x


class ConvMLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels * 4, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels * 4, in_channels, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ChooseInput(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_input))

    def forward(self, x):
        x = torch.einsum("bhwcn,n->bhwc", x, torch.sigmoid(self.w.to(x)))
        return x

class ChooseOP(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.w = nn.Parameter(torch.empty(5))
        self.op_list = nn.ModuleList([
            Conv3x3Act(model_dim, model_dim),
            Conv1x1Act(model_dim, model_dim),
            nn.AvgPool2d(3, 1, 1),
            ChannelAttention(model_dim),
            ConvMLP(model_dim)
        ])

    def forward(self, x):
        x_list = []
        for op in self.op_list:
            x_list.append(op(x))
        x = torch.stack(x_list, dim=-1)
        x = torch.einsum("bhwcn,n->bhwc", x, torch.sigmoid(self.w.to(x)))
        return x


class Node(nn.Module):
    def __init__(self, model_dim, num_input):
        super().__init__()
        self.choose = ChooseInput(num_input)
        self.op = ChooseOP(model_dim)

    def forward(self, x):
        x = self.choose(x)
        x = self.op(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, model_channels):
        super().__init__()
        self.nodes = nn.ModuleList([
            Node(model_channels, 11+i) for i in range(5)
        ])

    def forward(self, x, r1, r2, r3, r4, r5, d1, d2, d3, d4, d5):
        feature_list = torch.stack([x, r1, r2, r3, r4, r5, d1, d2, d3, d4, d5], dim=-1)
        for node in self.nodes:
            x = node(feature_list)
            feature_list = torch.cat([feature_list, x[..., None]], dim=-1)
        return x


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear_ = nn.Linear(model_channels, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear_(x)
        return x


class Model(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            model_channels,
            num_blocks,
            patch_size,
            use_checkpoint=True,
            use_xformer=True,
            dtype=torch.float32
    ):
        super(Model, self).__init__()

        self.dtype = dtype
        self.to(dtype)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.model_channels = model_channels
        self.num_blocks = num_blocks

        # self.blocks = nn.ModuleList([
        #     MM_DiT_Block(
        #         model_channels,
        #         num_heads=16,
        #         use_checkpoint=use_checkpoint,
        #         use_xformer=use_xformer
        #     )for _ in range(num_blocks)
        # ])

        self.blocks = nn.ModuleList([
            FusionLayer(model_channels) for _ in range(num_blocks)
        ])

        self.rgb_net = RGBNet(n_class=2)
        self.depth_net = DepthNet(n_class=2)
        self.input_embeddings = nn.ModuleList([
            nn.Linear(i, model_channels) for i in [64, 128, 256, 512, 512] * 2
        ])

        self.first_layer_img = FirstLayer(3, model_channels, patch_size=patch_size)
        self.first_layer_depth = FirstLayer(3, model_channels, patch_size=patch_size)
        self.first_layer_seg = FirstLayer(3, model_channels, patch_size=patch_size)

        self.final_layer = nn.ModuleList([
            FinalLayer(model_channels, 64, patch_size=patch_size) for i in range(num_blocks)
        ])

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 16, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, 1, 1)
            ) for i in range(num_blocks)
        ])
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.load_state()

    def unpatchify(self, x, h, w, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """

        x = x.reshape(shape=(x.shape[0], h//p, w//p, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, h, w))
        return x

    def pad_feature(self, x, max_length):
        length = max_length - x.shape[1]
        zeros = torch.zeros([1, length, self.model_channels]).repeat(x.shape[0], 1, 1).to(x)
        return torch.cat([x, zeros], dim=1)

    def update_network(self, data):
        out_list = []
        loss_list = []
        img = data["img"].permute([0, 3, 1, 2])
        depth = data["depth"].permute([0, 3, 1, 2])
        seg = data["seg"].permute([0, 3, 1, 2])

        bs, c, h, w = img.shape

        tgt = data["tgt"].permute([0, 3, 1, 2])
        tgt = torch.mean(tgt, dim=1, keepdim=True)

        rgb_list = list(self.rgb_net(img))
        d_list = list(self.depth_net(depth))[1:]

        feature_list = rgb_list + d_list
        for i, feature in enumerate(feature_list):
            bs, c, _, _ = feature.shape
            feature = torch.nn.functional.interpolate(feature, size=[64, 64]).permute(0, 2, 3, 1)
            feature_list[i] = self.input_embeddings[i](feature).permute(0, 3, 1, 2)

        rgb_list = feature_list[:5]
        d_list = feature_list[5:]

        x1 = self.first_layer_seg(img)
        x2 = self.first_layer_depth(depth)
        x3 = self.first_layer_seg(seg)

        x1 = x3
        x1 = x1.transpose(1, 2).reshape(d_list[0].shape)

        for index, block in enumerate(self.blocks):
            x1 = block(*([x1] + rgb_list + d_list))

        bs, c, _, _ = x1.shape
        x = self.final_layer[index](x1.reshape(bs, c, -1).transpose(1, 2))
        x = self.unpatchify(x, h, w, self.patch_size)
        x = self.conv[index](x)
        loss = self.loss(x, tgt) + self.iou_loss(x, tgt)
        out_list.append(x)
        loss_list.append(loss)

        # print(torch.softmax(self.weight[:, :, :1, :].reshape(4, 4), dim=1).detach().cpu().numpy())
        # print(torch.softmax(self.weight[:, :, 1:, :].reshape(4, 4), dim=1).detach().cpu().numpy())
        return sum(loss_list) / self.num_blocks, self.sigmoid(x), tgt, self.sigmoid(x), self.sigmoid(x)

    def update_arch(self, data):
        img = data["img"].permute([0, 3, 1, 2])
        depth = data["depth"].permute([0, 3, 1, 2])
        seg = data["seg"].permute([0, 3, 1, 2])

        tgt = data["tgt"].permute([0, 3, 1, 2])
        tgt = torch.mean(tgt, dim=1, keepdim=True)

        rgb_list = self.rgb_swin(img)[:4]
        depth_list = self.depth_swin(depth)[:4]
        feature_list = rgb_list + depth_list

        for i, feature in enumerate(feature_list):
            bs, c, h, w = feature.shape
            feature = torch.nn.functional.interpolate(feature, size=[96, 96]).reshape(bs, c, -1).transpose(1, 2)
            feature_list[i] = self.input_embeddings[i](feature)

        bs, c, h, w = img.shape
        x1 = self.first_layer_img(img)
        x2 = self.first_layer_depth(depth)
        x3 = self.first_layer_seg(seg)

        # for block in self.dit_blocks:
        #     x1, x2, x3 = block(x1, x2, x3)
        # bs, l, c = x1.shape
        # x = torch.stack([x1, x2, x3], dim=0)
        # x = torch.einsum("in,nblc->iblc", torch.softmax(self.w, dim=-1), x).reshape(bs, l, c)
        # feature_list = torch.stack(feature_list + [x3], dim=1)
        # feature_list = torch.cat(
        #     [self.cls_tokens_.repeat(bs, 1, 1).reshape(bs, feature_list.shape[1], 1, -1), feature_list], dim=2)

        # for i in range(self.num_blocks):
        #     bs, n, l, c = feature_list.shape
        #     cls_tokens = feature_list[:, :, 0, :]
        #     q, k = self.qk(cls_tokens).chunk(2, dim=-1)
        #     score = torch.einsum("blc,bci->bli", q, k.transpose(1, 2))[:, -1, :-1]
        #     score = torch.softmax(score, dim=-1)
        #     input_feature = torch.einsum("bin,bnl->bil", score.reshape(bs, 1, n-1),
        #                                  feature_list[:, :-1, :, :].reshape(bs, n-1, -1))
        #     input_feature = input_feature.reshape(bs, l, c)
        #     out1, out2 = self.single_blocks[i](input_feature, feature_list[:, -1, :, :])
        #     feature_list = torch.cat([feature_list, out2.reshape(bs, 1, l, c)], dim=1)

        x = self.final_layer(x3)
        x = self.unpatchify(x, h, w, 4)
        x = self.conv(x)
        return self.loss(x, tgt) + self.iou_loss(x, tgt), self.sigmoid(x), tgt, seg / 2 + 0.5

    def forward(self, data, mode):
        if mode == "network":
            return self.update_network(data)
        else:
            return self.update_arch(data)

    def iou_loss(self, pred, mask):
        pred = torch.sigmoid(pred)
        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        return iou.mean()

    def get_arch_params(self):
        params = []
        for i in self.qk.parameters():
            params.append(i)

        params.append(self.cls_tokens_)
        return params

    def get_network_params(self):
        params = []

        for i in self.first_layer_img.parameters():
            params.append(i)

        for i in self.first_layer_depth.parameters():
            params.append(i)

        for i in self.first_layer_seg.parameters():
            params.append(i)

        for i in self.input_embeddings.parameters():
            params.append(i)

        for i in self.final_layer.parameters():
            params.append(i)

        for i in self.blocks.parameters():
            params.append(i)

        for i in self.conv.parameters():
            params.append(i)

        params.append(self.weight)
        return params

    def load_state(self):
        self.rgb_net.load_state_dict(
            torch.load(
                r"D:\Code\python\CV\NAS_SOD\Models\DMRA\checkpoints\snapshot_iter_1000000.pth",
                map_location="cpu"
            )
        )

        self.depth_net.load_state_dict(
            torch.load(
                r"D:\Code\python\CV\NAS_SOD\Models\DMRA\checkpoints\depth_snapshot_iter_1000000.pth",
                map_location="cpu"
            )
        )

