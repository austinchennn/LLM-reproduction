"""评估: 困惑度、BLEU (自己实现的 corpus BLEU-4), 按源句长度分组的 BLEU

用法:
    python evaluate.py                                   # 用 <out_dir>/best.pt 在 test 集上评估, beam=5
    python evaluate.py --beam_size 1                     # greedy
    python evaluate.py --out_dir output_noattn           # 评估无 attention 的 baseline
"""

import argparse
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from config import Config
from data import PAD, encode_pairs, iter_batches


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """hyps / refs: [[token]], 每句一个参考译文。返回 0~100

    BLEU = BP · exp( (1/4) Σ_n log p_n )
      p_n: 整个语料上 n-gram 的截断精确率 (每个 n-gram 的计数不超过它在参考中出现的次数)
      BP : 简短惩罚, 译文总长 c < 参考总长 r 时为 exp(1 - r/c)
    """
    match, total = [0] * max_n, [0] * max_n
    c = r = 0
    for h, ref in zip(hyps, refs):
        c, r = c + len(h), r + len(ref)
        for n in range(1, max_n + 1):
            hc, rc = _ngrams(h, n), _ngrams(ref, n)
            match[n - 1] += sum(min(v, rc[g]) for g, v in hc.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    if c == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return 100 * bp * math.exp(log_p)


@torch.no_grad()
def evaluate_loss(model, data, batch_size, device):
    """按词平均的交叉熵 (pad 不计), 困惑度 = exp(loss)"""
    model.eval()
    total, count = 0.0, 0
    for src, src_len, tgt_in, tgt_out in iter_batches(data, batch_size, shuffle=False):
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits, _ = model(src, src_len, tgt_in)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                                 ignore_index=PAD, reduction="sum").item()
        count += (tgt_out != PAD).sum().item()
    loss = total / count
    return loss, math.exp(loss)


@torch.no_grad()
def translate_all(model, pairs, src_vocab, tgt_vocab, device, beam_size, len_alpha, batch_size=128):
    """pairs: [(src tokens, tgt tokens)] -> 译文 tokens 列表。beam_size=1 时按 batch 做 greedy"""
    from translate import beam_search, greedy_decode

    model.eval()
    data = encode_pairs(pairs, src_vocab, tgt_vocab)
    hyps = []
    if beam_size <= 1:
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            T = max(len(s) for s, _ in chunk)
            src = torch.full((len(chunk), T), PAD, dtype=torch.long)
            for b, (s, _) in enumerate(chunk):
                src[b, :len(s)] = torch.tensor(s)
            src_len = torch.tensor([len(s) for s, _ in chunk])
            out = greedy_decode(model, src.to(device), src_len, 2 * T + 2)
            hyps += [tgt_vocab.decode(row) for row in out.tolist()]
    else:
        for s, _ in data:
            src = torch.tensor([s], device=device)
            ids, _ = beam_search(model, src, torch.tensor([len(s)]), beam_size, len_alpha, 2 * len(s) + 2)
            hyps.append(tgt_vocab.decode(ids))
    return hyps


def load_pairs_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(col.split() for col in line.rstrip("\n").split("\t")) for line in f]


def main():
    from translate import load_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="默认 <out_dir>/best.pt")
    cfg, ns = Config.from_args(parser=p)
    device = torch.device(cfg.device)
    model, src_vocab, tgt_vocab, train_cfg = load_checkpoint(ns.ckpt or os.path.join(cfg.out_dir, "best.pt"),
                                                             device)
    test = load_pairs_tsv(os.path.join(cfg.out_dir, "test.tsv"))
    print(f"test 句对: {len(test):,}   attention: {not train_cfg.no_attention}")

    loss, ppl = evaluate_loss(model, encode_pairs(test, src_vocab, tgt_vocab), 128, device)
    print(f"test loss {loss:.3f}   perplexity {ppl:.2f}")

    hyps = translate_all(model, test, src_vocab, tgt_vocab, device, cfg.beam_size, cfg.len_alpha)
    refs = [t for _, t in test]
    print(f"test BLEU (beam={cfg.beam_size}): {corpus_bleu(hyps, refs):.2f}")

    # 按源句长度分组: 句子越长, attention 相比固定向量的优势应该越大 (论文 Figure 2)
    print("\n按源句长度分组的 BLEU:")
    for lo, hi in [(1, 4), (5, 7), (8, 10), (11, 15)]:
        idx = [i for i, (s, _) in enumerate(test) if lo <= len(s) <= hi]
        if idx:
            b = corpus_bleu([hyps[i] for i in idx], [refs[i] for i in idx])
            print(f"  {lo:>2}-{hi:<2} 词  n={len(idx):>5}  BLEU {b:.2f}")

    print("\n样例:")
    for (s, t), h in list(zip(test, hyps))[:10]:
        print(f"  fr : {' '.join(s)}\n  ref: {' '.join(t)}\n  hyp: {' '.join(h)}\n")


if __name__ == "__main__":
    main()
