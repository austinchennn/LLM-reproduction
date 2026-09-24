"""数据: 读语料 -> 建词表 -> 转 id -> 下采样 / 噪声表 -> 生成训练 batch"""

import os
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import torch

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def load_corpus(path, max_tokens=None):
    if not os.path.exists(path) and os.path.basename(path) == "text8":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        zip_path = path + ".zip"
        print(f"下载 text8 -> {zip_path}")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(os.path.dirname(path))
    with open(path, encoding="utf-8") as f:
        tokens = f.read().lower().split()
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


class Vocab:
    """词表: 过滤低频词 (min_count), 按词频降序编号"""

    def __init__(self, tokens, min_count):
        counter = Counter(tokens)
        words = [w for w, c in counter.most_common() if c >= min_count]
        self.itos = words
        self.stoi = {w: i for i, w in enumerate(words)}
        self.counts = np.array([counter[w] for w in words], dtype=np.int64)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        # 低频词 (OOV) 直接丢弃, 与原版 word2vec 一致
        return np.array([self.stoi[t] for t in tokens if t in self.stoi], dtype=np.int64)


def subsample_keep_prob(counts, t):
    """
    高频词下采样 (Mikolov 2013): "the", "of" 这类词出现太多、信息量少,
    按概率丢弃, 既加速训练又提升低频词的向量质量.
    这里用 word2vec C 源码里的公式: keep = (sqrt(f/t) + 1) * t / f, f 为词频占比
    """
    if t <= 0:
        return np.ones(len(counts))
    f = counts / counts.sum()
    return np.minimum(1.0, (np.sqrt(f / t) + 1) * t / f)


def build_noise_table(counts, table_size=int(1e7), power=0.75):
    """
    负采样的噪声分布 P_n(w) ∝ count(w)^0.75
    0.75 次方会把高频词概率压低、低频词抬高 (论文经验值).
    预先展开成一张大表, 采样时只需 randint 取下标, 比每步 multinomial 快得多.
    """
    p = counts.astype(np.float64) ** power
    p /= p.sum()
    reps = np.maximum(1, np.round(p * table_size)).astype(np.int64)
    return np.repeat(np.arange(len(counts)), reps)


def iter_batches(ids, mode, window, batch_size, keep_prob, pad_idx, rng, chunk=1_000_000):
    """
    分块生成训练样本 (避免一次性把几千万个 pair 放进内存), 块内打乱.
    - 下采样: 每个 epoch 重新随机丢一次高频词
    - 动态窗口: 每个中心词的实际窗口 b ~ U[1, window], 等价于给近处的词更高权重
    Skip-gram 产出 (center, context) 对; CBOW 产出 (context[2W] 带 padding, center)
    """
    ids = ids[rng.random(len(ids)) < keep_prob[ids]]
    n_total = len(ids)
    offsets = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])  # (2W,)

    for start in range(0, n_total, chunk):
        pos = np.arange(start, min(start + chunk, n_total))
        b = rng.integers(1, window + 1, size=len(pos))
        ctx_pos = pos[:, None] + offsets[None, :]  # (n, 2W)
        valid = (np.abs(offsets)[None, :] <= b[:, None]) & (ctx_pos >= 0) & (ctx_pos < n_total)
        ctx = np.where(valid, ids[np.clip(ctx_pos, 0, n_total - 1)], pad_idx)
        center = ids[pos]

        if mode == "sg":
            row, col = np.nonzero(valid)
            x, y = center[row], ctx[row, col]
        else:
            keep = valid.any(1)
            x, y = ctx[keep], center[keep]

        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]
        for i in range(0, len(x), batch_size):
            yield torch.from_numpy(x[i:i + batch_size]), torch.from_numpy(y[i:i + batch_size])
