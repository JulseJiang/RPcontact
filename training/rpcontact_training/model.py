import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 辅助函数：RoPE 旋转逻辑
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # ESM-2 风格的旋转位置嵌入应用
    return (x * cos) + (rotate_half(x) * sin)


class ESM2StyleAttention(nn.Module):
    def __init__(self, d_attn, n_head, dropout=0.1, max_len=5000):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_attn // n_head
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_attn, d_attn)
        self.k_proj = nn.Linear(d_attn, d_attn)
        self.v_proj = nn.Linear(d_attn, d_attn)
        self.out_proj = nn.Linear(d_attn, d_attn)
        self.dropout = nn.Dropout(dropout)

        # 预计算旋转频率 (inv_freq)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq)

    def _get_rotary_emb(self, seq_len, device):
        # 动态生成适合当前长度的旋转矩阵
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, query, key, value):
        batch_size, L_q, _ = query.shape
        L_k = key.shape[1]

        # 投影与分头
        q = self.q_proj(query).view(batch_size, L_q, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, L_k, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, L_k, self.n_head, self.d_head).transpose(1, 2)

        # 获取并应用旋转位置嵌入 (RoPE)
        cos_q, sin_q = self._get_rotary_emb(L_q, query.device)
        cos_k, sin_k = self._get_rotary_emb(L_k, query.device)

        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)

        # 计算注意力分数 (此时内积已包含相对位置信息)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attn_weights), v)

        context = context.transpose(1, 2).contiguous().view(batch_size, L_q, -1)
        return self.out_proj(context)


class TwoTrackAttention(nn.Module):
    def __init__(self, d_attn, n_head, d_ff=512, dropout=0.1) -> None:
        super().__init__()
        # 替换为 ESM-2 风格的旋转注意力
        self.self_attn = ESM2StyleAttention(d_attn, n_head, dropout)
        self.cross_attn = ESM2StyleAttention(d_attn, n_head, dropout)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_attn)

        self.ff = nn.Sequential(
            nn.Linear(d_attn, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_attn)
        )
        self.norm2 = nn.LayerNorm(d_attn)
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, obj_update, obj_message):
        s_out = self.self_attn(obj_update, obj_update, obj_update)
        c_out = self.cross_attn(obj_update, obj_message, obj_message)

        obj_update = obj_update + self.dropout_self(s_out) + self.dropout_cross(c_out)
        obj_update = self.norm1(obj_update)

        ff_out = self.ff(obj_update)
        obj_update = obj_update + self.dropout_final(ff_out)
        obj_update = self.norm2(obj_update)
        return obj_update


class SymertricTwoTrackAttention(nn.Module):
    def __init__(self, d_attn, n_head, d_ff=512, dropout=0.1, sync=False) -> None:
        super().__init__()
        self.tta1 = TwoTrackAttention(d_attn, n_head, d_ff, dropout)
        self.tta2 = TwoTrackAttention(d_attn, n_head, d_ff, dropout)
        self.sync = sync

    def forward(self, obj_1, obj_2):
        if self.sync:
            return self.tta1(obj_1, obj_2), self.tta2(obj_2, obj_1)
        else:
            obj_1 = self.tta1(obj_1, obj_2)
            obj_2 = self.tta2(obj_2, obj_1)
            return obj_1, obj_2


class LinearFF(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1) -> None:
        super().__init__()
        self.emb = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 统一使用 GELU

    def forward(self, f_in):
        # 保持你原有的投影逻辑
        return self.norm(self.dropout(self.activation(self.emb(f_in))))


class ProteinRNAInteraction_rela(nn.Module):
    def __init__(self, d_pro, d_rna, n_layers, d_attn, n_head=4, d_ff=512, dropout=0.1, sync=False) -> None:
        super().__init__()
        self.pro_emb = LinearFF(d_pro, d_attn)
        self.rna_emb = LinearFF(d_rna, d_attn)

        self.layers = nn.ModuleList([
            SymertricTwoTrackAttention(d_attn, n_head, d_ff, dropout, sync=sync)
            for _ in range(n_layers)
        ])

        # 预测头优化
        self.feature_fuse = nn.LayerNorm(d_attn)
        self.pred = nn.Linear(d_attn, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_pro, f_rna):
        # 1. 维度调整与投影
        f_pro = self.pro_emb(f_pro.permute(0, 2, 1))
        f_rna = self.rna_emb(f_rna.permute(0, 2, 1))

        # 2. 演化层（RoPE 已在内部处理相对位置）
        for layer in self.layers:
            f_pro, f_rna = layer(f_pro, f_rna)

        # 3. 构造交互平面 [B, L_pro, L_rna, D]
        f_pro_sq = f_pro.unsqueeze(2)
        f_rna_sq = f_rna.unsqueeze(1)

        # 4. 特征融合（使用 Hadamard 积 + LayerNorm）
        interaction = self.feature_fuse(f_pro_sq * f_rna_sq)

        prob = self.sigmoid(self.pred(interaction))
        return prob