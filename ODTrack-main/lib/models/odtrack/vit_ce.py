import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock
from collections import deque

_logger = logging.getLogger(__name__)


class FIFOQueue:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def set_max_size(self, new_size):
        if new_size <= 0:
            raise ValueError("queue must > 0")

        self.max_size = new_size
        while len(self.queue) > new_size:
            self.queue.popleft()

    def __iter__(self):
        return iter(self.queue)

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            raise IndexError("queue is empty")

    def peek(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            raise IndexError("queue is empty")

    def peek_last(self):
        if len(self.queue) > 0:
            return self.queue[-1]
        else:
            raise IndexError("queue is empty")

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def size(self):
        return len(self.queue)

    def __str__(self):
        return f"FIFOQueue(max_size={self.max_size}, current_size={len(self.queue)}, items={list(self.queue)})"

    def clear(self):
        self.queue = deque(maxlen=self.max_size)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

        # gr
        self.size_template = 5
        self.size_search = 5
        self.q_template = FIFOQueue(self.size_template)
        self.q_online = FIFOQueue(self.size_template)
        self.q_search = FIFOQueue(self.size_search)
        self.dim = 576
        self.channel = 768
        self.dist_map = torch.ones([self.dim]).cuda()
        self.T_U = None
        self.T_miu = None
        self.T_sigma = None
        self.S_U = []
        self.S_miu = []
        self.S_sigma = []
        self.frame = 0
        self.time = 0
        self.k = 3

    def compute_template_subsapce(self, q_template, subspace_num, k):
        combined_z = torch.cat(list(q_template), dim=-1)
        T_miu = torch.mean(combined_z.squeeze(0), dim=-1)


        U, S, V = torch.linalg.svd(combined_z.squeeze(0))
        T_U = U[:, :k]
        T_sigma = torch.diag(S[:k])

        return T_miu, T_U, T_sigma

    def compute_search_subspace(self, q_search, subspace_num, T_miu, T_U, k):
        combined_x = torch.stack(list(q_search)[-subspace_num:], dim=2).squeeze(0).transpose(1, 2)
        S_miu, S_U, S_sigma, dist_list = [], [], [], []

        for i in range(combined_x.shape[0]):
            matrix = combined_x[i]
            miu_xi = torch.mean(matrix, dim=-1)
            S_miu.append(miu_xi)


            Ux, Sx, Vx = torch.linalg.svd(matrix)
            U_xi = Ux[:, :k]
            Sx_diag_k = torch.diag(Sx[:k])

            S_U.append(U_xi)
            S_sigma.append(Sx_diag_k)

        return S_miu, S_U, S_sigma

    def search_subspace_distance(self, q_search, subspace_num, T_miu, T_U, k):
        combined_x = torch.stack(list(q_search)[-subspace_num:], dim=0).squeeze(1).transpose(0, 2)
        S_miu, S_U, S_sigma, dist_list = [], [], [], []

        for i in range(combined_x.shape[0]):
            matrix = combined_x[i]
            miu_xi = torch.mean(matrix, dim=-1)
            S_miu.append(miu_xi)


            Ux, Sx, Vx = torch.linalg.svd(matrix)
            U_xi = Ux[:, :k]
            Sx_diag_k = torch.diag(Sx[:k])

            S_U.append(U_xi)
            S_sigma.append(Sx_diag_k)

            distGi = torch.norm(T_U @ T_U.T - U_xi @ U_xi.T, 'fro')
            distMi = (T_miu - miu_xi).unsqueeze(0) @ (
                    2 * torch.eye(768).cuda() - T_U @ T_U.T - U_xi @ U_xi.T
            ) @ (T_miu - miu_xi)

            disti = 0.9999 * distGi + 0.0001 * distMi
            # disti = distGi
            dist_list.append(disti)

        dist_map = nn.functional.softmax(-torch.stack(dist_list, dim=0), dim=0) * self.dim
        dist_list.clear()
        return S_miu, S_U, S_sigma, dist_map

    def search_subspace_distance_sin(self, q_search, subspace_num, T_miu, T_U, k):
        combined_x = torch.stack(list(q_search)[-subspace_num:], dim=0).squeeze(1).transpose(0, 2)
        S_miu, S_U, S_sigma, dist_list = [], [], [], []

        for i in range(combined_x.shape[0]):
            matrix = combined_x[i]
            miu_xi = torch.mean(matrix, dim=-1)
            S_miu.append(miu_xi)


            Ux, Sx, Vx = torch.linalg.svd(matrix)
            U_xi = Ux[:, :k]
            Sx_diag_k = torch.diag(Sx[:k])

            S_U.append(U_xi)
            S_sigma.append(Sx_diag_k)

            distPi = self.principal_angles(T_U, U_xi)
            distMi = (T_miu - miu_xi).unsqueeze(0) @ (
                    2 * torch.eye(768).cuda() - T_U @ T_U.T - U_xi @ U_xi.T
            ) @ (T_miu - miu_xi)

            # disti = 0.999 * distPi + 0.001 * distMi
            disti = distPi
            dist_list.append(disti)

        dist_map = nn.functional.softmax(-torch.stack(dist_list, dim=0), dim=0) * self.dim
        dist_list.clear()
        return S_miu, S_U, S_sigma, dist_map

    def incremental_template_subspace(self, q_online, subspace_num, T_U, T_sigma, T_miu, forget_factor, k):

        B_z = torch.cat(list(q_online)[-subspace_num:], dim=-1).squeeze(0)
        pro_z = B_z - torch.matmul(torch.matmul(T_U, T_U.T), B_z)
        Q_z, R_z = torch.linalg.qr(pro_z, mode='reduced')

        R_upper_z = torch.cat((forget_factor * T_sigma, torch.matmul(T_U.T, B_z)), dim=1)
        Orth_z = torch.matmul(Q_z.T, pro_z)
        zero_matrix_z = torch.zeros(Orth_z.shape[0], T_sigma.shape[1]).cuda()
        R_lower_z = torch.cat((zero_matrix_z, Orth_z), dim=1)
        R_z = torch.cat((R_upper_z, R_lower_z), dim=0)

        u_z, s_z, v = torch.linalg.svd(R_z)
        u_z = u_z[:, :k]
        uz_update = torch.matmul(torch.cat((T_U, Q_z), dim=1), u_z)

        T_U_new = uz_update
        T_sigma_new = torch.diag(s_z[:k])
        T_miu_new = (torch.mean(B_z, dim=-1) + T_miu) / 2

        return T_U_new, T_sigma_new, T_miu_new

    def incremental_search_subspace(self, q_search, subspace_num, S_U, S_sigma, S_miu, T_U, T_miu, r, k):
        dist_list = []


        B1 = torch.stack(list(q_search)[-subspace_num:], dim=0).squeeze(1).transpose(0, 2)

        for i in range(B1.shape[0]):
            a = B1[i] - torch.matmul(torch.matmul(S_U[i], S_U[i].T), B1[i])
            Q, R = torch.linalg.qr(a, mode='reduced')


            R_upper = torch.cat((r * S_sigma[i], torch.matmul(S_U[i].T, B1[i])), dim=1)
            Orth = torch.matmul(Q.T, a)
            zero_matrix = torch.zeros(Orth.shape[0], S_sigma[i].shape[1]).cuda()
            R_lower = torch.cat((zero_matrix, Orth), dim=1)


            R = torch.cat((R_upper, R_lower), dim=0)
            u, s, v = torch.linalg.svd(R)
            u = u[:, :k]


            us_update = torch.matmul(torch.cat((S_U[i], Q), dim=1), u)
            S_sigma[i] = torch.diag(s[:k])
            S_U[i] = us_update


            mius_update = (torch.mean(B1[i], dim=-1) + S_miu[i]) / 2
            S_miu[i] = mius_update


            distgi = torch.norm(T_U @ T_U.T - us_update @ us_update.T, 'fro')
            distmi = (T_miu - mius_update).unsqueeze(0) @ (
                        2 * torch.eye(768).cuda() - T_U @ T_U.T - us_update @ us_update.T) @ (
                             T_miu - mius_update)
            dist_i = 0.9999 * distgi + 0.0001 * distmi
            # dist_i = distgi
            dist_list.append(dist_i)

        dist = torch.stack(dist_list, dim=0)
        dist_map = nn.functional.softmax(-dist, dim=0) * self.dim
        dist_list.clear()

        return S_U, S_sigma, S_miu, dist_map

    def incremental_search_subspace_sin(self, q_search, subspace_num, S_U, S_sigma, S_miu, T_U, T_miu, r, k):
        dist_list = []

        B1 = torch.stack(list(q_search)[-subspace_num:], dim=0).squeeze(1).transpose(0, 2)

        for i in range(B1.shape[0]):
            a = B1[i] - torch.matmul(torch.matmul(S_U[i], S_U[i].T), B1[i])
            Q, R = torch.qr(a)

            R_upper = torch.cat((r * S_sigma[i], torch.matmul(S_U[i].T, B1[i])), dim=1)
            Orth = torch.matmul(Q.T, a)
            zero_matrix = torch.zeros(Orth.shape[0], S_sigma[i].shape[1]).cuda()
            R_lower = torch.cat((zero_matrix, Orth), dim=1)

            R = torch.cat((R_upper, R_lower), dim=0)
            u, s, v = torch.linalg.svd(R)
            u = u[:, :k]

            us_update = torch.matmul(torch.cat((S_U[i], Q), dim=1), u)
            S_sigma[i] = torch.diag(s[:k])
            S_U[i] = us_update

            mius_update = (torch.mean(B1[i], dim=-1) + S_miu[i]) / 2
            S_miu[i] = mius_update

            distpi = self.principal_angles(T_U, us_update)
            distmi = (T_miu - mius_update).unsqueeze(0) @ (
                        2 * torch.eye(768).cuda() - T_U @ T_U.T - us_update @ us_update.T) @ (
                             T_miu - mius_update)
            # dist_i = 0.999 * distpi + 0.001 * distmi
            dist_i = distpi
            dist_list.append(dist_i)

        dist = torch.stack(dist_list, dim=0)
        dist_map = nn.functional.softmax(-dist, dim=0) * self.dim
        dist_list.clear()

        return S_U, S_sigma, S_miu, dist_map

    def principal_angles(self, u1, u2):

        T = torch.mm(u1.t(), u2).float()
        Pu, Sigma, Pv = torch.linalg.svd(T)  # , compute_uv=False
        cos_theta = Sigma
        sin_theta_2 = 1. - pow(cos_theta, 2)
        sin_theta_2 = torch.where(sin_theta_2 < 0, torch.zeros_like(sin_theta_2), sin_theta_2)
        sin_theta_2_sum = sin_theta_2.sum()
        return sin_theta_2_sum

    def forward_features_gr(self, z, x, mask_z=None, mask_x=None,
                            ce_template_mask=None, ce_keep_rate=None,
                            return_last_attn=False, track_query=None,
                            token_type="add", token_len=1, time=None, r_s=None,
                            r_t=None, update=None, s_num=None, t_num=None, sub_template=None, show_sub=None
                            ):
        self.frame += 1
        self.time -= 1

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)

        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)

        sub_template = torch.stack(sub_template, dim=1)
        sub_template = sub_template.flatten(0, 1)
        sub_template = self.patch_embed(sub_template)

        # if len(show_sub) == t_num:
        #     show_sub = torch.stack(show_sub, dim=1)
        #     show_sub = show_sub.flatten(0, 1)
        #     show_sub = self.patch_embed(show_sub)

        max_pool = nn.MaxPool1d(kernel_size=144, stride=144)

        if len(sub_template) == t_num and self.frame <= t_num + 1:
            sub_reshaped = sub_template

            for i in range(sub_reshaped.shape[0]):
                z_i = sub_reshaped[i].unsqueeze(0).permute(0, 2, 1)
                z_pooled = max_pool(z_i.detach())  # (1, 768, 1)
                self.q_template.enqueue(z_pooled)

        if (self.frame - 1) % update == 0 and self.frame != 1:
            o_reshaped = sub_template

            for i in range(o_reshaped.shape[0]):
                o_i = o_reshaped[i].unsqueeze(0).permute(0, 2, 1)
                o_pooled = max_pool(o_i.detach())  # (1, 768, 1)
                self.q_online.enqueue(o_pooled)
            self.T_U, self.T_sigma, self.T_miu = self.incremental_template_subspace(
                self.q_online, t_num, self.T_U, self.T_sigma, self.T_miu, r_t, self.k
            )
            if self.q_search.is_full():
                self.S_U, self.S_sigma, self.S_miu, self.dist_map = self.incremental_search_subspace(
                    self.q_search, s_num, self.S_U, self.S_sigma, self.S_miu, self.T_U, self.T_miu, r_s, self.k
                )
                self.time = time
            self.q_online.clear()
        if self.T_U is None and self.q_template.size() == t_num:
            # if self.T_U is None and target.size() == subspace_num:
            # for i in target:
            #     t1 = self.patch_embed(i)
            #     z1 = max_pool(t1.permute(0, 2, 1).detach())
            #     self.q_template.enqueue(z1)
            self.T_miu, self.T_U, self.T_sigma = self.compute_template_subsapce(self.q_template, t_num, self.k)
        if self.q_search.size() == s_num and not self.S_U:
            self.S_miu, self.S_U, self.S_sigma, self.dist_map = self.search_subspace_distance(
                self.q_search, s_num, self.T_miu, self.T_U, self.k)
            self.time = time
            # self.S_miu, self.S_U, self.S_sigma = self.compute_search_subspace(
            #     self.q_search, subspace_num, self.T_miu, self.T_U, self.k)
        self.q_search.enqueue(x.permute(0, 2, 1))
        if self.time == 0:
            self.dist_map = torch.ones([self.dim]).cuda()
        x = self.dist_map.view(1, self.dim, 1) * x
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query is None:
                    query = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query.size(1)
                    new_query = self.cls_token.expand(B, token_len - track_len, -1)
                    query = torch.cat([new_query, track_query], dim=1)
            elif token_type == "add":
                new_query = self.cls_token.expand(B, token_len, -1)  # copy B times
                query = new_query if track_query is None else track_query + new_query
            query = query + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)

        lens_z = z.shape[1]  # HW
        lens_x = x.shape[1]  # HW

        x = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)  # (B, 1+z+x, 768)
            query_len = query.size(1)
        x = self.pos_drop(x)

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            if self.add_cls_token:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)
            else:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        if self.add_cls_token:
            query = x[:, :query_len]
            z = x[:, query_len:lens_z_new + query_len]
            x = x[:, lens_z_new + query_len:]
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([query, z, x], dim=1)

        # aux_dict = {}
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)

        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query is None:
                    query = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query.size(1)
                    new_query = self.cls_token.expand(B, token_len - track_len, -1)
                    query = torch.cat([new_query, track_query], dim=1)
            elif token_type == "add":
                new_query = self.cls_token.expand(B, token_len, -1)  # copy B times
                query = new_query if track_query is None else track_query + new_query
            query = query + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)

        lens_z = z.shape[1]  # HW
        lens_x = x.shape[1]  # HW

        x = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)  # (B, 1+z+x, 768)
            query_len = query.size(1)
        x = self.pos_drop(x)

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            if self.add_cls_token:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)
            else:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        if self.add_cls_token:
            query = x[:, :query_len]
            z = x[:, query_len:lens_z_new + query_len]
            x = x[:, lens_z_new + query_len:]
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([query, z, x], dim=1)

        # aux_dict = {}
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None,
                token_type="add", token_len=1):
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            track_query=track_query, token_type=token_type, token_len=token_len)
        return x, aux_dict

    def forward_gr(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                   tnc_keep_rate=None, return_last_attn=False, track_query=None,
                   token_type="add", token_len=1, time=None, r_s=None, r_t=None, update=None, s_num=None, t_num=None,
                   sub_template=None, show_sub=None):
        x, aux_dict = self.forward_features_gr(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                               track_query=track_query, token_type=token_type, token_len=token_len,
                                               time=time, r_s=r_s, r_t=r_t, update=update, s_num=s_num,
                                               t_num=t_num, sub_template=sub_template, show_sub=show_sub)
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print('Load pretrained model from: ' + pretrained)
            except:
                print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
