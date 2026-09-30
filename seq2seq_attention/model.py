"""Bahdanau 2014: 双向 GRU 编码器 + 加性注意力 + GRU 解码器 (maxout 输出层)

形状记号: B = batch, T = 源句长, m = 目标句长, E = emb_dim, H = hid_dim, A = attn_dim, V = 词表大小
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import PAD


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.init_state = nn.Linear(hid_dim, hid_dim)  # s_0 = tanh(W_s · ←h_1)

    def forward(self, src, src_len):
        """返回 annotations [B,T,2H], 解码器初始状态 s_0 [B,H], 无 attention 时用的固定上下文 [B,2H]"""
        x = self.drop(self.embed(src))
        # pack 让 GRU 跳过 pad, 后向 GRU 才能从真正的最后一个词开始读; lengths 必须在 CPU 上
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h_n = self.rnn(packed)
        ann, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # h_n: [2,B,H]; h_n[0] 是前向读完 x_T 的状态, h_n[1] 是后向读完 x_1 的状态 (即 ←h_1)
        s0 = torch.tanh(self.init_state(h_n[1]))
        fixed_ctx = torch.cat([h_n[0], h_n[1]], dim=-1)
        return ann, s0, fixed_ctx


class BahdanauAttention(nn.Module):
    """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j),  α_ij = softmax_j(e_ij),  c_i = Σ_j α_ij h_j"""

    def __init__(self, hid_dim, attn_dim):
        super().__init__()
        self.W_a = nn.Linear(hid_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(2 * hid_dim, attn_dim)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def precompute(self, ann):
        """U_a h_j 与解码步无关, 整句只算一次: [B,T,A]"""
        return self.U_a(ann)

    def forward(self, s_prev, keys, ann, mask):
        # s_prev [B,H], keys [B,T,A], ann [B,T,2H], mask [B,T] (True 为真实词)
        e = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + keys)).squeeze(-1)  # [B,T]
        e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)                                              # [B,T]
        ctx = torch.bmm(alpha.unsqueeze(1), ann).squeeze(1)                       # [B,2H]
        return ctx, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, attn_dim, maxout_dim, dropout,
                 use_attention=True, tie_weights=False):
        super().__init__()
        self.use_attention = use_attention
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attn = BahdanauAttention(hid_dim, attn_dim) if use_attention else None
        self.cell = nn.GRUCell(emb_dim + 2 * hid_dim, hid_dim)
        # t_i = maxout(W_o [s_i ; E y_{i-1} ; c_i]): 先映射到 2M 维, 相邻两维取 max
        self.pre_out = nn.Linear(hid_dim + emb_dim + 2 * hid_dim, 2 * maxout_dim)
        self.out = nn.Linear(maxout_dim, vocab_size)
        if tie_weights:
            self.out.weight = self.embed.weight
        self.drop = nn.Dropout(dropout)

    def step(self, y_prev, s_prev, enc):
        """解码一步。y_prev [B], s_prev [B,H], enc 为 Seq2Seq.encode 的输出
        返回 logits [B,V], s_i [B,H], alpha [B,T] (无 attention 时为 None)"""
        emb = self.drop(self.embed(y_prev))
        if self.use_attention:
            ctx, alpha = self.attn(s_prev, enc["keys"], enc["ann"], enc["mask"])  # 用 s_{i-1} 查询
        else:
            ctx, alpha = enc["fixed_ctx"], None
        s = self.cell(torch.cat([emb, ctx], dim=-1), s_prev)
        t = self.pre_out(torch.cat([s, emb, ctx], dim=-1))
        t = t.view(t.size(0), -1, 2).max(dim=-1).values                        # maxout, [B,M]
        logits = self.out(self.drop(t))
        return logits, s, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, cfg.emb_dim, cfg.hid_dim, cfg.dropout)
        self.decoder = Decoder(tgt_vocab_size, cfg.emb_dim, cfg.hid_dim, cfg.attn_dim, cfg.maxout_dim,
                               cfg.dropout, use_attention=not cfg.no_attention,
                               tie_weights=cfg.tie_weights)
        self.hid_dim = cfg.hid_dim
        self._init_weights()

    def _init_weights(self):
        """循环权重正交初始化 (论文做法), 其余权重 N(0, 0.01²), 偏置为 0"""
        H = self.hid_dim
        for name, p in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif "weight_hh" in name:
                for k in range(0, p.size(0), H):          # r, z, n 三个门的 [H,H] 分块各自正交
                    nn.init.orthogonal_(p.data[k:k + H])
            else:
                nn.init.normal_(p, std=0.01)
        with torch.no_grad():
            self.encoder.embed.weight[PAD].zero_()
            self.decoder.embed.weight[PAD].zero_()

    def encode(self, src, src_len):
        ann, s0, fixed_ctx = self.encoder(src, src_len)
        enc = {"ann": ann, "mask": src != PAD, "fixed_ctx": fixed_ctx}
        enc["keys"] = self.decoder.attn.precompute(ann) if self.decoder.use_attention else None
        return enc, s0

    def forward(self, src, src_len, tgt_in, teacher_forcing=1.0):
        """训练用: 返回 logits [B,m,V] 和注意力权重 [B,m,T] (无 attention 时为 None)"""
        enc, s = self.encode(src, src_len)
        y = tgt_in[:, 0]
        logits_all, alphas = [], []
        for i in range(tgt_in.size(1)):
            logits, s, alpha = self.decoder.step(y, s, enc)
            logits_all.append(logits)
            alphas.append(alpha)
            if i + 1 < tgt_in.size(1):
                # teacher forcing: 用真实的下一个输入; 否则用模型自己这一步的预测
                use_gold = teacher_forcing >= 1.0 or random.random() < teacher_forcing
                y = tgt_in[:, i + 1] if use_gold else logits.argmax(-1)
        attn = torch.stack(alphas, dim=1) if alphas[0] is not None else None
        return torch.stack(logits_all, dim=1), attn
