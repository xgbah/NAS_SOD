import copy

import cv2
import numpy as np

import xformers
import os
from xformers.ops import RMSNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.DiffudionModels.DiffusionModel.ldm.modules.diffusionmodules.util import checkpoint


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

        q = q.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).reshape(bs * num_heads, n, -1).contiguous()
        k = k.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).reshape(bs * num_heads, n, -1).contiguous()
        v = v.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).reshape(bs * num_heads, n, -1).contiguous()

        q = self.norm_q(q).to(v)
        k = self.norm_k(k).to(v)

        score = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(c//num_heads)
        attn_map = attn = torch.softmax(score, dim=-1)

        attn = torch.bmm(attn, v)

        attn = attn.reshape(bs, num_heads, n, c // num_heads).transpose(1, 2).reshape(bs, n, c)

        return self.proj(attn), attn_map


class Conv3x3Act(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        bs, l, c = x.shape
        h = w = int(np.sqrt(l-1))
        cls_token = x[:, :1, :]
        x = x[:, 1:, :].transpose(1, 2).reshape(bs, c, h, w)
        x = self.act(self.c(x)).reshape(bs, c, -1).transpose(1, 2)
        return torch.cat([cls_token, x], dim=1)


class Conv1x1Act(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        bs, l, c = x.shape
        h = w = int(np.sqrt(l-1))
        cls_token = x[:, :1, :]
        x = x[:, 1:, :].transpose(1, 2).reshape(bs, c, h, w)
        x = self.act(self.c(x)).reshape(bs, c, h*w).transpose(1, 2)
        return torch.cat([cls_token, x], dim=1)


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


class StaticInput(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_input, 1))

    def forward(self, x):
        index = torch.argmax(self.w, dim=0)
        x = x[:, :, :, index]
        return x

class StaticOP(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 1))
        self.op_list = nn.ModuleList([
            Conv1x1Act(model_dim, model_dim),
            Conv3x3Act(model_dim, model_dim),
            nn.Identity(),
            # AttnBlock(model_dim, num_heads=16, xformer=False),
            MLP(model_dim, model_dim*4)
        ])

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )

    def _forward(self, x):
        index = torch.argmax(self.w, dim=0)
        x = self.op_list[index](x)
        # print("operation", torch.softmax(self.w.to(x), dim=0).reshape(-1).float().detach().cpu().numpy())
        return x


class StaticNode(nn.Module):
    def __init__(self, model_dim, num_input):
        super().__init__()
        self.choose = StaticInput(num_input)
        self.op = StaticOP(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.attn = AttnBlock(model_dim, 16)

    def forward(self, x):
        x = self.choose(x)
        res = x
        x, attn_map = self.attn(self.norm1(x))
        x = x + res
        x = self.op(self.norm2(x)) + x
        return x, attn_map


class FusionBlock(nn.Module):
    def __init__(self, model_channels):
        super().__init__()
        self.nodes = nn.ModuleList([
            StaticNode(model_channels, 1+i) for i in range(4)
        ])

    def forward(self, x):
        feature_list = x[..., None]
        for node in self.nodes:
            x, attn_map = node(feature_list)
            feature_list = torch.cat([feature_list, x[..., None]], dim=-1)
        return x, attn_map


class DecoderBlock(nn.Module):
    def __init__(self, model_channels, num_heads):
        super().__init__()
        self.attn = AttnBlock(model_channels, num_heads=num_heads, xformer=True)
        self.mlp = MLP(model_channels, model_channels * 4)
        self.norm1 = nn.LayerNorm(model_channels)
        self.norm2 = nn.LayerNorm(model_channels)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )

    def _forward(self, x):
        res = x
        x, attn = self.attn(self.norm1(x))
        x = x + res
        x = self.mlp(self.norm2(x)) + x
        return x


class Predict_head(nn.Module):
    def __init__(self, model_channels, hidden_dim, predict_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(model_channels),
            MLP(model_channels, hidden_dim),
            nn.LayerNorm(model_channels),
            MLP(model_channels, hidden_dim),
            nn.LayerNorm(model_channels),
            MLP(model_channels, hidden_dim),
        )
        self.proj = nn.Linear(model_channels, predict_dim)

    def forward(self, x):
        return self.proj(self.head(x))

class Encoder(nn.Module):
    def __init__(self, model_channels, patch_size, num_blocks, predict_dim, num_heads=16, mask_rate=0.75):
        super().__init__()
        self.mask_rate = mask_rate
        self.patch_size = patch_size
        self.num_heads = 16
        self.first_layer = FirstLayer(patch_size=patch_size, in_channels=3, model_channels=model_channels)
        self.blocks = nn.ModuleList([
            FusionBlock(model_channels) for _ in range(num_blocks)
        ])
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_channels))

        self.cls_head = Predict_head(model_channels, 256, predict_dim)
        self.patch_head = Predict_head(model_channels, 256, predict_dim)

    def mask_img(self, img):
        bs, num_patches, c = img.shape
        num_masks = int((1 - self.mask_rate) * num_patches)

        noise = torch.randn(bs, num_patches)
        _, index = torch.sort(noise, dim=-1)
        _, restored_index = torch.sort(index, dim=-1)

        index = index.reshape(bs, num_patches, 1).repeat(1, 1, c)
        index = index[:, :num_masks, :].to(img.device)
        restored_index = restored_index.reshape(bs, num_patches, 1).repeat(1, 1, c).to(img.device)

        masked_image = img.gather(1, index)
        return masked_image, restored_index

    def forward(self, x):
        x = self.first_layer(x)
        bs, l, c = x.shape
        cls_token = self.cls_token.repeat(bs, 1, 1).to(x)
        x = torch.cat([cls_token, x], dim=1)
        out_list = [x]
        for block in self.blocks:
            x, attn_map = block(x)
            out_list.append(x)

        output_dict = {
            "cls_tokens": self.cls_head(x[:, 0, :]),
            "patch_tokens": self.patch_head(x[:, 1:, :]),
            "latent": out_list[-1],
            "attn_map": attn_map
        }
        return output_dict


class Decoder(nn.Module):
    def __init__(self, model_channels, patch_size, num_blocks):
        super().__init__()
        self.patch_size = patch_size
        self.final_layer = FinalLayer(patch_size=patch_size, out_channels=64, model_channels=model_channels)
        self.blocks = nn.ModuleList([
            DecoderBlock(model_channels, num_heads=16) for _ in range(num_blocks)
        ])

        self.conv = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def unpatchify(self, x, h, w, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """

        x = x.reshape(shape=(x.shape[0], h, w, -1, p, p))
        x = torch.einsum('bhwcpq->bhpwqc', x)
        x = x.reshape(shape=(x.shape[0], h*p, w*p, -1)).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        bs, l, c = x.shape
        h = w = int(np.sqrt(l))
        pose_embed = torch.from_numpy(get_2d_sincos_pos_embed(c, [h, w])).to(x)[None, ...]
        x = x + pose_embed
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        h = w = int(np.sqrt(x.shape[1]))
        x = self.unpatchify(x, h, w, self.patch_size)
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, model_channels, patch_size, num_blocks, mask_rate=0.75, mask_size=64, predict_dim=2048):
        super().__init__()
        self.patch_size = patch_size
        self.mask_rate = mask_rate
        self.mask_size = mask_size
        self.s_encoder = Encoder(model_channels, patch_size, num_blocks, predict_dim=predict_dim)
        self.t_encoder = Encoder(model_channels, patch_size, num_blocks, predict_dim=predict_dim)
        self.t_encoder.load_state_dict(self.s_encoder.state_dict())

        self.decoder = Decoder(model_channels, patch_size, num_blocks=6)

        self.c = torch.zeros(1, predict_dim, requires_grad=False)
        self.c_p = torch.zeros(1, predict_dim, requires_grad=False)
        self.w = nn.Parameter(torch.ones(1, 1, model_channels))
        self.c_moment = 0.9

    def generate_mask(self, h, w, mask_rate, num_masks=1):
        num_patches = h * w
        num_mask = int(mask_rate * num_patches)

        if num_masks == 1:
            mask = np.array([0] * num_mask + [1] * (num_patches - num_mask))
            np.random.shuffle(mask)
            masks = mask.reshape(1, 1, h, w)
        else:
            masks = []
            for i in range(num_masks):
                mask = np.array([0] * num_mask + [1] * (num_patches - num_mask))
                np.random.shuffle(mask)
                mask = mask.reshape(1, 1, h, w)
                masks.append(mask)
            masks = np.concatenate(masks, axis=0)
        return masks


    def dino_loss(self, t_out, s_out, c):
        t_out = torch.softmax((t_out - c) / 0.01, dim=-1)
        s_out = torch.softmax(s_out / 0.1, dim=-1)
        loss = 0
        cnt = 0
        for index_s, s in enumerate(s_out):
            for index_t, t in enumerate(t_out):
                loss += -1 * (t * torch.log(s)).sum(dim=-1).mean()
                cnt += 1
        return loss / cnt

    def cos_loss(self, t_out, s_out, c):
        t_out = torch.softmax((t_out - c) / 0.01, dim=-1)
        s_out = torch.softmax(s_out / 0.1, dim=-1)
        loss = 0
        cnt = 0
        for index_s, s in enumerate(s_out):
            for index_t, t in enumerate(t_out):
                if index_s == index_t:
                    loss_ = (t * s).sum(dim=-1) / torch.sqrt((t**2).sum(dim=-1)) / torch.sqrt((s**2).sum(dim=-1))
                else:
                    loss_ = -1 * (t * s).sum(dim=-1) / torch.sqrt((t**2).sum(dim=-1)) / torch.sqrt((s**2).sum(dim=-1))
                loss += loss_.mean()
                cnt += 1
        return loss / cnt

    def iBot_loss(self, t_out, s_out, c, mask):  # t_out[num_images, num_patches, c]
        t_out = torch.softmax((t_out - c) / 0.01, dim=-1)
        s_out = torch.softmax(s_out / 0.1, dim=-1)
        num_images, _, h, w = mask.shape
        mask = mask.reshape(num_images, 1, h*w).reshape(num_images, h*w)
        loss = -1 * (t_out * torch.log(s_out)).sum(dim=-1).mean()  # loss[num_images, num_patches]
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def pad_tokens(self, tokens, h, w, p):
        bs, l, c = tokens.shape
        num_zeros = h*w // p**2 - l
        return torch.cat([tokens, self.w.repeat(bs, num_zeros, 1).to(tokens)], dim=1)

    def forward(self, data):
        local_images = data["local_images"]
        global_images = data["global_images"]
        num_local_images = len(local_images)
        num_global_images = len(global_images)
        bs, h_local, w_local, c = local_images[0].shape
        bs, h_global, w_global, c = global_images[0].shape
        p = self.patch_size

        local_images = torch.stack(local_images, dim=0).reshape(-1, h_local, w_local, c).permute(0, 3, 1, 2)
        global_images = torch.stack(global_images, dim=0).reshape(-1, h_global, w_global, c).permute(0, 3, 1, 2)
        total_images = torch.cat([global_images, local_images], dim=0)
        num_images, _, img_h, img_w = total_images.shape

        h, w = img_h // p, img_w // p

        mask = self.generate_mask(h, w, self.mask_rate, num_masks=num_local_images)
        mask = torch.from_numpy(mask).to(local_images)
        img_mask = mask.reshape(num_local_images, 1, h, 1, w, 1).repeat(1, 1, 1, p, 1, p).reshape(num_local_images, 1, h * p, w * p)
        masked_img_s = local_images * img_mask

        s_output = self.s_encoder(masked_img_s)
        cls_tokens_s, patche_tokens_s, latent_s, attn = s_output["cls_tokens"], s_output["patch_tokens"], s_output["latent"], s_output["attn_map"]

        empty_tokens = (1 - mask.reshape(num_local_images, 1, h * w).transpose(1, 2)) * self.w.to(latent_s)
        unmasked_tokens = latent_s[:, 1:, :] * mask.reshape(num_local_images, 1, h * w).transpose(1, 2)

        latent_s = empty_tokens + unmasked_tokens
        re_constructed_s = self.decoder(latent_s)

        with torch.no_grad():
            t_output = self.t_encoder(global_images)
            cls_tokens_t, patche_tokens_t = t_output["cls_tokens"], t_output["patch_tokens"]

            self.c = self.c * self.c_moment + (1-self.c_moment) * cls_tokens_t.mean(dim=0, keepdim=True)
            _, _, c = patche_tokens_t.shape
            self.c_p = self.c_p * self.c_moment + (1 - self.c_moment) * patche_tokens_t.reshape(-1, c).mean(dim=0, keepdim=True)

        loss_dino = self.dino_loss(
            cls_tokens_t.reshape(num_global_images, bs, -1),
            cls_tokens_s.reshape(num_local_images, bs, -1),
            self.c
        )

        # loss_ibot = self.iBot_loss(
        #     patche_tokens_t,
        #     patche_tokens_s,
        #     self.c_p,
        #     mask
        # )

        loss_mae = ((re_constructed_s - local_images) ** 2).mean()

        attn_map = attn[:, 0, 1:].reshape(bs, -1, h*w).mean(dim=1).reshape(bs, 1, h, w)
        attn_map = nn.functional.interpolate(attn_map, [h_global, w_global])
        return loss_mae + loss_dino, local_images / 2 + 0.5, masked_img_s / 2 + 0.5, re_constructed_s / 2 + 0.5, attn_map

    def EMA(self, decay=0.996):
        with torch.no_grad():
            for model_t_v, model_s_v in zip(self.t_encoder.state_dict().values(), self.s_encoder.state_dict().values()):
                model_t_v.copy_(decay * model_t_v + (1 - decay) * model_s_v)
