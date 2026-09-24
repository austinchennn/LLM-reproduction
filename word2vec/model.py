"""Word2Vec 模型: 两套 embedding + 负采样损失, 支持 Skip-gram / CBOW"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    """
    两套向量:
      in_embed  (v): 中心词/输入侧, 训练完一般拿它当词向量
      out_embed (u): 上下文/输出侧, 只在训练时用来算 logits
    负采样目标 (对每个正样本):
      loss = -log σ(u_o · v) - Σ_{k=1..K} log σ(-u_k · v),   u_k ~ P_n(w)
    """

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.pad_idx = vocab_size  # CBOW 上下文的 padding 位, 不参与训练
        # sparse=True: 每步只有 batch 里出现过的那几行有梯度, 对大词表很关键
        self.in_embed = nn.Embedding(vocab_size + 1, dim, padding_idx=self.pad_idx, sparse=True)
        self.out_embed = nn.Embedding(vocab_size, dim, sparse=True)
        # 与原版一致的初始化: 输入侧小均匀分布, 输出侧全 0
        nn.init.uniform_(self.in_embed.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_embed.weight)
        with torch.no_grad():
            self.in_embed.weight[self.pad_idx].zero_()

    def _neg_sampling_loss(self, h, target, neg):
        # h: (B, D) 输入侧表示; target: (B,) 正样本; neg: (B, K) 负样本
        u_pos = self.out_embed(target)                    # (B, D)
        u_neg = self.out_embed(neg)                       # (B, K, D)
        pos_score = (h * u_pos).sum(-1)                   # (B,)
        neg_score = torch.bmm(u_neg, h.unsqueeze(2)).squeeze(2)  # (B, K)
        # logsigmoid 比 log(sigmoid(x)) 数值更稳定
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1))
        return loss.mean()

    def forward_sg(self, center, context, neg):
        # Skip-gram: 用中心词预测上下文词
        return self._neg_sampling_loss(self.in_embed(center), context, neg)

    def forward_cbow(self, context, center, neg):
        # CBOW: 上下文词向量取平均 (忽略 padding) 预测中心词
        mask = (context != self.pad_idx).unsqueeze(-1).float()  # (B, 2W, 1)
        h = (self.in_embed(context) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self._neg_sampling_loss(h, center, neg)

    def embeddings(self):
        return self.in_embed.weight[:-1].detach()
