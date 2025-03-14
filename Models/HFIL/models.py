import cv2
import numpy as np

import xformers
import os
from xformers.ops import RMSNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.HFIL.SwinTransformer import SwinTransformer, LFF
from Models.MAE.model import Encoder


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
        x = x + pose_embed
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
    def __init__(self, dim, num_heads, norm=nn.LayerNorm):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = norm(dim//num_heads)
        self.norm_k = norm(dim//num_heads)
        self.num_heads = num_heads

    def forward(self, x1, x2):
        q = self.q(x1)
        k, v = self.kv(x2).chunk(2, dim=-1)
        bs, n, c = q.shape
        num_heads = self.num_heads

        q = q.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).contiguous()
        k = k.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).contiguous()
        v = v.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).contiguous()
        q = self.norm_q(q).to(v)
        k = self.norm_k(k).to(v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        attn = attn.transpose(1, 2).reshape(bs, n, c)
        return attn


class CMFM(nn.Module):
    def __init__(self, c_dim, use_checkpoint=True):
        super(CMFM, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.fc1 = nn.Linear(c_dim, c_dim)
        self.fc2 = nn.Linear(c_dim, c_dim)
        self.gate = nn.Sigmoid()
        self.weight = nn.Parameter(torch.empty(2))

    def forward(self, x1, x2):
        x1 = x1 + self.gate(self.fc1(x2)) * x2
        x2 = x2 + self.gate(self.fc2(x1)) * x1
        return self.weight[0]*x1 + self.weight[1]*x2


class DecoderBlock(nn.Module):
    def __init__(self, c_dim, num_heads, use_checkpoint=False):
        super(DecoderBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(c_dim)
        self.fusion = CMFM(c_dim)
        self.mlp = MLP(c_dim, c_dim * 4)

    def forward(self, x1, x2):
        x = self.fusion(x1, x2)
        x = self.mlp(self.layer_norm(x)) + x
        return x


class FirstLayer(nn.Module):
    def __init__(self, in_channels, model_channels, patch_size=2, use_checkpoint=False):
        super().__init__()
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.model_channels = model_channels
        self.patch_emb = PatchEmbed(in_channels, model_channels, patch_size=patch_size)

    def forward(self, x):
        x = self.patch_emb(x)
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


class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = self.load_encoder()
        self.d_encoder = self.load_encoder()

    def load_encoder(self):
        from Models.SwinTransformer.models.swin_transformer_v2 import SwinTransformerV2
        model = SwinTransformerV2(
            img_size=256,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=16,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            use_checkpoint=True,
            pretrained_window_sizes=[12, 12, 12, 6]
        )
        state_dict = torch.load(
            r".\Models\SwinTransformer\models\swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth",
            map_location="cpu"
        )["model"]
        model.load_state_dict(state_dict)
        print("Swin Transformer ckpt Loaded")
        return model

    def forward(self, img, depth):
        rgb_list = self.rgb_encoder(img)[:4]
        d_list = self.d_encoder(depth)[:4]
        return rgb_list + d_list

class DINOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = self.load_encoder()
        self.d_encoder = self.load_encoder()

    def load_encoder(self):
        from Models.DINOv2.vision_transformer import vit_g as vit
        model = vit()
        state_dict = torch.load(
            r".\Models\DINOv2\dinov2_vitl14_reg4_pretrain.pth",
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        print("DINOv2 ckpt Loaded")
        return model

    def forward(self, img, depth):
        img = nn.functional.interpolate(img, [518, 518])
        depth = nn.functional.interpolate(depth, [518, 518])
        rgb_list = self.rgb_encoder(img, is_training=True)
        d_list = self.d_encoder(depth, is_training=True)
        return rgb_list[::6] + d_list[::6]


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = self.load_encoder()
        self.d_encoder = self.load_encoder()

    def load_encoder(self):
        from Models.HFIL.resnet import ResNet, Bottleneck
        encoder = ResNet(Bottleneck, [3, 8, 36, 3])
        state_dict = torch.load(r".\Models\HFIL\resnet152.pth", map_location='cpu')
        encoder.load_state_dict(state_dict, strict=True)
        print("resnet152 encoder loaded")
        return encoder

    def forward(self, img, depth):
        r1, r2, r3, r4 = self.rgb_encoder(img)
        d1, d2, d3, d4 = self.rgb_encoder(depth)
        feature_list = [r1, r2, r3, r4, d1, d2, d3, d4]
        for i in range(len(feature_list)):
            bs, c, h, w = feature_list[i].shape
            # print(feature_list[i].shape)
            feature_list[i] = feature_list[i].reshape(bs, c, -1).transpose(-1, -2)
        return feature_list


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

        self.blocks = nn.ModuleList([
            DecoderBlock(model_channels, 16, True) for _ in range(num_blocks)
        ])

        self.encoder = DINOEncoder()

        self.input_embeddings = nn.ModuleList([
            nn.Linear(i, model_channels) for i in [1024, 1024, 1024, 1024] * 2
        ])

        self.first_layer_img = FirstLayer(3, model_channels, patch_size=patch_size)
        self.first_layer_depth = FirstLayer(3, model_channels, patch_size=patch_size)
        self.first_layer_seg = FirstLayer(3, model_channels, patch_size=patch_size)

        self.final_layer = FinalLayer(model_channels, 64, patch_size=patch_size)

        self.conv = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.qk = nn.Linear(model_channels, model_channels*2)
        self.cls_token = nn.Parameter(torch.empty(1, 4*2 + 1, 1, model_channels))
        self.w = nn.Parameter(torch.empty(2))

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
        img = data["image"].permute([0, 3, 1, 2])
        depth = data["depth"].permute([0, 3, 1, 2])
        seg = data["seg"].permute([0, 3, 1, 2])

        bs, c, h, w = img.shape

        tgt = data["tgt"].permute([0, 3, 1, 2])
        tgt = torch.mean(tgt, dim=1, keepdim=True)

        feature_list = self.encoder(img, depth)
        for i, feature in enumerate(feature_list):
            bs, n, c = feature.shape
            h_f = w_f = int(np.sqrt(n))
            feature = feature.reshape(bs, h_f, w_f, c).permute(0, 3, 1, 2)
            feature = nn.functional.interpolate(feature, [h // self.patch_size, w // self.patch_size])
            feature = feature.reshape(bs, c, -1).permute(0, 2, 1)
            feature_list[i] = self.input_embeddings[i](feature)
            # feature_list[i] = feature

        x1 = self.first_layer_img(img)
        x2 = self.first_layer_depth(depth)
        # x3 = self.first_layer_seg(seg)

        weight = torch.softmax(self.w.to(x1), dim=-1)
        x1 = weight[0] * x1 + weight[1] * x2
        feature_list = torch.stack(feature_list + [x1], dim=1)
        feature_list = torch.cat(
            [self.cls_token.repeat(bs, 1, 1, 1), feature_list], dim=2)

        for i in range(self.num_blocks):
            bs, n, l, c = feature_list.shape
            cls_tokens = feature_list[:, :, 0, :]
            q, k = self.qk(cls_tokens).chunk(2, dim=-1)
            score = torch.einsum("bnc,bcm->bnm", q, k.transpose(1, 2))[:, -1, :-1]
            score = torch.softmax(score, dim=-1)
            input_feature = torch.einsum("bnm,bmi->bni", score.reshape(bs, 1, n-1),
                                         feature_list[:, :-1, :, :].reshape(bs, n-1, -1))
            input_feature = input_feature.reshape(bs, l, c)
            out = self.blocks[i](input_feature, feature_list[:, -1, :, :])
            feature_list = torch.cat([feature_list, out.reshape(bs, 1, l, c)], dim=1)

        x = self.final_layer(out[:, 1:, :])
        x = self.unpatchify(x, h, w, self.patch_size)
        x = self.conv(x)
        loss = self.loss(x, tgt) + self.iou_loss(x, tgt)

        return loss, self.sigmoid(x), tgt, self.sigmoid(x), self.sigmoid(x)

    def update_network_without_nas(self, data):
        img = data["image"].permute([0, 3, 1, 2])
        depth = data["depth"].permute([0, 3, 1, 2])

        bs, c, h, w = img.shape

        tgt = data["tgt"].permute([0, 3, 1, 2])
        tgt = torch.mean(tgt, dim=1, keepdim=True)

        feature_list = self.encoder(img, depth)
        for i, feature in enumerate(feature_list):
            bs, n, c = feature.shape
            h_f = w_f = int(np.sqrt(n))
            feature = feature.reshape(bs, h_f, w_f, c).permute(0, 3, 1, 2)
            feature = nn.functional.interpolate(feature, [h // self.patch_size, w // self.patch_size])
            feature = feature.reshape(bs, c, -1).permute(0, 2, 1)
            feature_list[i] = self.input_embeddings_[i](feature)

        rgb_list = feature_list[:4]
        d_list = feature_list[4:]

        x1 = self.first_layer_img(img)
        x2 = self.first_layer_depth(depth)

        weight = torch.softmax(self.w.to(x1), dim=-1)
        x1 = weight[0] * x1 + weight[1] * x2

        f1, r1, d1 = self.blocks[0](x1, rgb_list[0], d_list[0])
        f2, r2, d2 = self.blocks[1](f1, rgb_list[1], d_list[1])
        f3, r3, d3 = self.blocks[2](f2, rgb_list[2], d_list[2])
        f4, r4, d4 = self.blocks[3](f3, rgb_list[3], d_list[3])

        f, r5, d5 = self.blocks[4](f4, r4, d4)
        f, r5, d5 = self.blocks[5](f, r3, d3)
        f, r5, d5 = self.blocks[6](f, r2, d2)
        f, r5, d5 = self.blocks[7](f, r1, d1)

        x = self.final_layer(f)
        x = self.unpatchify(x, h, w, self.patch_size)
        x = self.conv(x)
        loss = self.loss(x, tgt) + self.iou_loss(x, tgt)

        return loss, self.sigmoid(x), tgt, self.sigmoid(x), self.sigmoid(x)

    def forward(self, data):
        return self.update_network(data)

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

        for i in self.input_embeddings_.parameters():
            params.append(i)

        for i in self.final_layer.parameters():
            params.append(i)

        for i in self.blocks.parameters():
            params.append(i)

        for i in self.conv.parameters():
            params.append(i)

        for i in self.qk.parameters():
            params.append(i)

        params.append(self.cls_token)

        params.append(self.w)
        return params

