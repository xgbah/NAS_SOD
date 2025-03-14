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
        self.act = nn.ReLU()
        self.patch_embed_ = nn.Conv2d(in_channels=in_channels,
                                     out_channels=emb_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     padding=0)
        self.norm = norm_layer(emb_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.patch_embed_(x)
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
        attn_map = attn_map.reshape(bs, num_heads, n, n).mean(dim=1)[:, 0, 1:]
        attn = torch.bmm(attn, v)

        attn = attn.reshape(bs, num_heads, n, c // num_heads).transpose(1, 2).reshape(bs, n, c)

        return self.proj(attn), attn_map

class CrossAttn(nn.Module):
    def __init__(self, dim, num_heads, norm=nn.LayerNorm, xformer=False):
        super().__init__()
        self.qkv_x = nn.Linear(dim, dim * 3)
        self.proj_x = nn.Linear(dim, dim)
        self.norm_q_x = norm(dim//num_heads)
        self.norm_k_x = norm(dim//num_heads)

        self.qkv_y = nn.Linear(dim, dim * 3)
        self.proj_y = nn.Linear(dim, dim)
        self.norm_q_y = norm(dim // num_heads)
        self.norm_k_y = norm(dim // num_heads)

        self.num_heads = num_heads
        self.xformer = xformer

    def get_qkv(self, x, qkv, norm_q, norm_k):
        num_heads = self.num_heads

        q, k, v = qkv(x).chunk(3, dim=-1)
        bs, n, c = q.shape

        q = q.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).reshape(bs * num_heads, n, -1).contiguous()
        k = k.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).reshape(bs * num_heads, n, -1).contiguous()
        v = v.reshape(bs, n, num_heads, c // num_heads).transpose(1, 2).reshape(bs * num_heads, n, -1).contiguous()
        q = norm_q(q).to(v)
        k = norm_k(k).to(v)
        return q, k, v

    def forward(self, x, y):
        bs, n, c = x.shape
        num_heads = self.num_heads
        q_x, k_x, v_x = self.get_qkv(x, self.qkv_x, self.norm_q_x, self.norm_k_x)
        q_y, k_y, v_y = self.get_qkv(y, self.qkv_y, self.norm_q_y, self.norm_k_y)

        q = torch.cat([q_x, q_y], dim=1)
        k = torch.cat([k_x, k_y], dim=1)
        v = torch.cat([v_x, v_y], dim=1)

        score = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(c//num_heads)
        attn = torch.softmax(score, dim=-1)

        attn = torch.bmm(attn, v)

        attn = attn.reshape(bs, num_heads, -1, c // num_heads).transpose(1, 2).reshape(bs, -1, c)

        x = attn[:, :n, :]
        y = attn[:, n:, :]
        return self.proj_x(x), self.proj_y(y)


class Conv3x3Act(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        bs, l, c = x.shape
        h = w = int(np.sqrt(l))
        x = x.transpose(1, 2).reshape(bs, c, h, w)
        x = self.act(self.c(x)).reshape(bs, c, -1).transpose(1, 2)
        return x


class Conv1x1Act(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Linear(in_channels, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.c(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 16, in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(x)))
        out = max_out
        return self.sigmoid(out) * x


class ChooseInput(nn.Module):
    def __init__(self, num_input, debug=False):
        super().__init__()
        self.debug = debug
        self.w = nn.Parameter(torch.empty(num_input, 1))

    def forward(self, x):
        b, l, c, n = x.shape
        x = torch.einsum("blcn,ni->blci", x, torch.softmax(self.w.to(x)/0.01, dim=0)).reshape(b, l, c)
        if self.debug:
            print("input", torch.softmax(self.w.to(x)/0.01, dim=0).reshape(-1).float().detach().cpu().numpy())
        return x


class Node(nn.Module):
    def __init__(self, model_dim, num_input, debug):
        super().__init__()
        self.debug = debug
        self.w_ = nn.Parameter(torch.empty(4 + num_input, 1))
        self.attn = AttnBlock(model_dim, num_heads=16, xformer=False)
        self.op_list = nn.ModuleList([
            Conv1x1Act(model_dim, model_dim),
            ChannelAttention(model_dim),
            nn.Identity(),
            MLP(model_dim, model_dim * 4)
        ])

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )

    def _forward(self, x_list):
        new_list = []
        x = x_list[..., -1]
        for op in self.op_list:
            new_list.append(op(x))
        new_list = torch.stack(new_list, dim=-1)
        x = torch.cat([x_list, new_list], dim=-1)
        b, l, c, n = x.shape
        x = torch.einsum("blcn,ni->blci", x, torch.softmax(self.w_.to(x) / 0.01, dim=0)).reshape(b, l, c)
        x_c = x
        x, attn = self.attn(x)
        x = x + x_c
        if self.debug:
            print("operation", torch.softmax(self.w.to(x) / 0.01, dim=0).reshape(-1).float().detach().cpu().numpy())
        return x, attn


class FusionBlock(nn.Module):
    def __init__(self, model_channels, debug=False):
        super().__init__()
        self.nodes = nn.ModuleList([
            Node(model_channels, 1+i, debug=debug) for i in range(4)
        ])

    def forward(self, x):
        feature_list = x[..., None]
        for node in self.nodes:
            x, attn = node(feature_list)
            feature_list = torch.cat([feature_list, x[..., None]], dim=-1)
        return x, attn


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
        x = self.attn(self.norm1(x))
        x = x + res
        x = self.mlp(self.norm2(x)) + x
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # 防止分母计算为0
        self._eps = eps
        # 仿射变换参数，缩放和平移norm后的数据分布
        self._beta = nn.Parameter(torch.zeros(1, dim))
        self._gamma = nn.Parameter(torch.ones(1, dim))

    def forward(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)  # 计算均值
        var = input.var(dim=-1, unbiased=False, keepdim=True)  # 计算有偏方差
        input = (input - mean) / torch.sqrt(var + self._eps)  # 执行标准化
        return input * self._gamma.to(input) + self._beta.to(input)  # 仿射变换


class Predict_head(nn.Module):
    def __init__(self, model_channels, hidden_dim, bottle_neck_dim, predict_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottle_neck_dim),
            nn.LayerNorm(bottle_neck_dim),
            nn.Linear(bottle_neck_dim, hidden_dim),
        )

        self.proj = nn.Linear(hidden_dim, predict_dim)
        self.norm = LayerNorm(predict_dim, eps=1e-6)

    def forward(self, x):
        return self.proj(self.head(x))


class Encoder(nn.Module):
    def __init__(self, model_channels, patch_size, num_blocks, predict_dim, num_heads=16, mask_rate=0.75):
        super().__init__()
        self.mask_rate = mask_rate
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.first_layer = FirstLayer(patch_size=patch_size, in_channels=3, model_channels=model_channels)
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_channels))

        self.blocks = nn.ModuleList([
            FusionBlock(model_channels, debug=False) for _ in range(num_blocks)
        ])
        self.predict_head_ = Predict_head(model_channels, hidden_dim=2048, bottle_neck_dim=256, predict_dim=predict_dim)

    def pad_tokens(self, tokens, h, w, p, index):
        bs, l, c = tokens.shape
        num_zeros = h*w // p**2 - l
        zeros = torch.zeros(bs, num_zeros, c)
        tokens = torch.cat([tokens, zeros.to(tokens)], dim=1)
        return tokens.gather(dim=1, index=index)

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
        x = torch.cat([self.cls_token.to(x).repeat(bs, 1, 1), x], dim=1)

        if self.mask_rate > 0:
            masked_x, index_x = self.mask_img(x)
            out_list_x = [masked_x]
        else:
            index_x = None
            masked_x = x
            out_list_x = [masked_x]

        for i in range(self.num_blocks):
            masked_x, attn_x = self.blocks[i](masked_x)
            out_list_x.append(masked_x)

        cls_token = self.predict_head_(masked_x[:, 0, :])
        output_dict = {
            "out": out_list_x,
            "index": index_x,
            "attn": attn_x,
            "cls_token": cls_token
        }
        return output_dict


class Decoder(nn.Module):
    def __init__(self, model_channels, patch_size, num_blocks):
        super().__init__()
        self.patch_size = patch_size
        self.final_layer_ = FinalLayer(patch_size=patch_size, out_channels=64, model_channels=model_channels)
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
        x = self.final_layer_(x)
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
        self.encoder = Encoder(model_channels, patch_size, num_blocks, predict_dim=predict_dim, mask_rate=0)
        self.t_encoder = copy.deepcopy(self.encoder)
        self.register_buffer("c", torch.zeros(1, 1, predict_dim))
        self.c_moment = 0.9

        self.K_loss = KoLeoLoss()

    def dino_loss(self, t_out, s_out, c):
        t_out = torch.softmax((t_out - c) / 0.07, dim=-1)
        s_out = torch.softmax(s_out / 0.1, dim=-1)
        loss = 0
        cnt = 0
        for index_s, s in enumerate(s_out):
            for index_t, t in enumerate(t_out):
                loss += -1 * (t * torch.log(s)).sum(dim=-1).mean()
                cnt += 1
        return loss / cnt

    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        world_size = 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def infoNCE_loss(self, t_out, s_out):
        t_out = nn.functional.normalize(t_out, dim=-1)
        s_out = nn.functional.normalize(s_out, dim=-1)
        pos_score = (t_out @ s_out.T) / 0.05
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def forward(self, data):
        num_x = len(data["image_x"])
        num_y = len(data["image_y"])

        bs, img_h, img_w, c = data["image_x"][0].shape
        img_x = torch.stack(data["image_x"], dim=0).reshape(-1, img_h, img_w, c).permute(0, 3, 1, 2).to(self.c.dtype)
        img_y = torch.stack(data["image_y"], dim=0).reshape(-1, img_h, img_w, c).permute(0, 3, 1, 2).to(self.c.dtype)

        p = self.patch_size
        h, w = img_h // p, img_w // p

        enc_output = self.encoder(img_x)
        cls_x, attn_x = enc_output["cls_token"], enc_output["attn"]

        enc_output = self.encoder(img_y)
        cls_y, attn_y = enc_output["cls_token"], enc_output["attn"]


        # with torch.no_grad():
        #     self.EMA()
        #     t_output = self.t_encoder(img_y)
        #     cls_y, attn_y = t_output["cls_token"], t_output["attn"]
        #     cls_y = self.sinkhorn_knopp_teacher(cls_y, 0.07)
        #     self.c = self.c * self.c_moment + (1-self.c_moment) * cls_y.mean(dim=0, keepdim=True)
        loss = self.infoNCE_loss(cls_y, cls_x)
        # loss_dino = self.dino_loss(
        #     cls_y.reshape(num_y, bs, -1),
        #     cls_x.reshape(num_x, bs, -1),
        #     self.c
        # )
        loss_k = self.K_loss(cls_x) + self.K_loss(cls_y)
        # loss = loss_dino + sum(loss_k)

        attn_x = attn_x.reshape(num_x*bs, 1, h, w)
        attn_x = nn.functional.interpolate(attn_x, [img_h, img_w])

        attn_y = attn_y.reshape(num_y*bs, 1, h, w)
        attn_y = nn.functional.interpolate(attn_y, [img_h, img_w])
        return loss, img_x / 2 + 0.5, attn_x, img_y / 2 + 0.5, attn_y

    def EMA(self, decay=0.996):
        with torch.no_grad():
            for model_t_v, model_s_v in zip(self.t_encoder.state_dict().values(), self.encoder.state_dict().values()):
                model_t_v.copy_(decay * model_t_v + (1 - decay) * model_s_v)


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss

