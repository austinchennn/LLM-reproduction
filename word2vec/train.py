"""
训练入口

用法:
    python train.py --mode sg                         # 默认使用 data/text8, 不存在则自动下载
    python train.py --mode cbow --dim 100 --epochs 3
    python train.py --corpus my_corpus.txt --mode sg  # 自己的语料: 空格分词的纯文本 (中文请先用 jieba 分好词)
"""

import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from config import Config
from data import Vocab, build_noise_table, iter_batches, load_corpus, subsample_keep_prob
from evaluate import report
from model import Word2Vec


def save(model, vocab, cfg):
    """保存为 word2vec 文本格式, 可直接被 gensim KeyedVectors.load_word2vec_format 读取"""
    os.makedirs(cfg.out_dir, exist_ok=True)
    emb = model.embeddings().cpu().numpy()
    np.save(os.path.join(cfg.out_dir, "embeddings.npy"), emb)
    with open(os.path.join(cfg.out_dir, "vectors.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {emb.shape[1]}\n")
        for w, v in zip(vocab.itos, emb):
            f.write(w + " " + " ".join(f"{x:.6f}" for x in v) + "\n")
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print(f"已保存到 {cfg.out_dir}")


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # sparse embedding 的梯度在 MPS 上支持不完整, Mac 上直接用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(cfg, end="\n\n")

    # 数据
    t0 = time.time()
    tokens = load_corpus(cfg.corpus, cfg.max_tokens)
    vocab = Vocab(tokens, cfg.min_count)
    ids = vocab.encode(tokens)
    del tokens
    keep_prob = subsample_keep_prob(vocab.counts, cfg.subsample)
    noise_table = torch.from_numpy(build_noise_table(vocab.counts)).to(device)
    kept = keep_prob[ids].sum()
    print(f"tokens={len(ids):,}  vocab={len(vocab):,}  下采样后约 {kept:,.0f} tokens/epoch  "
          f"({time.time() - t0:.1f}s)", end="\n\n")

    # 模型 & 优化器
    model = Word2Vec(len(vocab), cfg.dim).to(device)
    # 稀疏梯度只能配 SparseAdam 或 SGD; SparseAdam 对学习率不敏感, 更省心
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=cfg.lr)

    # 学习率线性衰减 (原版 word2vec 的做法), 需要先估计总步数
    samples_per_epoch = kept * (cfg.window + 1) if cfg.mode == "sg" else kept  # 动态窗口平均每侧 (W+1)/2
    total_steps = max(1, int(math.ceil(samples_per_epoch / cfg.batch_size)) * cfg.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: max(cfg.min_lr_ratio, 1 - s / total_steps))

    forward = model.forward_sg if cfg.mode == "sg" else model.forward_cbow

    step, running = 0, 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_epoch = time.time()
        for x, y in iter_batches(ids, cfg.mode, cfg.window, cfg.batch_size, keep_prob, model.pad_idx, rng):
            x, y = x.to(device), y.to(device)
            # 负样本: 从噪声表里随机取; 偶尔撞到正样本本身, 原版也不处理, 影响可忽略
            neg = noise_table[torch.randint(len(noise_table), (len(y), cfg.neg), device=device)]
            loss = forward(x, y, neg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            running += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps}  loss {running / cfg.log_every:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t_epoch:.0f}s")
                running = 0.0
        print(f"== epoch {epoch} done ({time.time() - t_epoch:.0f}s)")
        report(model, vocab, cfg.probes, cfg.analogies)
        print()

    save(model, vocab, cfg)
    return model, vocab


if __name__ == "__main__":
    train(Config.from_args())
