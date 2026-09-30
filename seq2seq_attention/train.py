"""训练入口

用法:
    python train.py                                         # 默认数据 data/eng-fra.txt, 不存在则自动下载
    python train.py --max_pairs 5000 --epochs 2             # 快速调试
    python train.py --no_attention true --out_dir output_noattn   # 无 attention 的 baseline
"""

import json
import math
import os
import random
import time
from dataclasses import asdict

import torch
import torch.nn as nn

from config import Config
from data import PAD, Vocab, encode_pairs, iter_batches, load_pairs, split_pairs
from evaluate import corpus_bleu, evaluate_loss, translate_all
from model import Seq2Seq


def save_pairs_tsv(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for s, t in pairs:
            f.write(" ".join(s) + "\t" + " ".join(t) + "\n")


def train(cfg: Config):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(cfg)

    # 数据: 词表只用训练集建, valid/test 中没见过的词都是 <unk>
    pairs = load_pairs(cfg.data_path, cfg.max_len, cfg.max_pairs)
    train_pairs, valid_pairs, test_pairs = split_pairs(pairs, cfg.valid_ratio, cfg.test_ratio, cfg.seed)
    src_vocab = Vocab([s for s, _ in train_pairs], cfg.min_count)
    tgt_vocab = Vocab([t for _, t in train_pairs], cfg.min_count)
    train_data = encode_pairs(train_pairs, src_vocab, tgt_vocab)
    valid_data = encode_pairs(valid_pairs, src_vocab, tgt_vocab)
    save_pairs_tsv(test_pairs, os.path.join(cfg.out_dir, "test.tsv"))
    print(f"句对 train/valid/test = {len(train_pairs):,}/{len(valid_pairs):,}/{len(test_pairs):,}   "
          f"词表 fr={len(src_vocab):,} en={len(tgt_vocab):,}")

    # 模型 & 优化器: 编码器、attention、解码器在同一个优化器里端到端一起训练
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.1f}M   device={device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg.lr_patience)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)  # 对非 pad 的目标词取平均

    bleu_pairs = valid_pairs[:cfg.bleu_samples]
    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, run_loss, run_n = time.time(), 0.0, 0
        n_batches = math.ceil(len(train_data) / cfg.batch_size)
        for step, (src, src_len, tgt_in, tgt_out) in enumerate(
                iter_batches(train_data, cfg.batch_size, shuffle=True, rng=rng), 1):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits, _ = model(src, src_len, tgt_in, cfg.teacher_forcing)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()

            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if step % cfg.log_every == 0:
                avg = run_loss / run_n
                print(f"epoch {epoch} step {step}/{n_batches}  loss {avg:.3f}  ppl {math.exp(avg):.1f}  "
                      f"{(time.time() - t0) / step * 1000:.0f}ms/step")
                run_loss, run_n = 0.0, 0

        # 验证: 困惑度 + greedy BLEU + 几条样例
        val_loss, val_ppl = evaluate_loss(model, valid_data, 128, device)
        hyps = translate_all(model, bleu_pairs, src_vocab, tgt_vocab, device, beam_size=1, len_alpha=0)
        bleu = corpus_bleu(hyps, [t for _, t in bleu_pairs])
        scheduler.step(val_loss)
        print(f"== epoch {epoch}  valid loss {val_loss:.3f}  ppl {val_ppl:.2f}  BLEU(greedy) {bleu:.2f}  "
              f"lr {optimizer.param_groups[0]['lr']:.1e}  {time.time() - t0:.0f}s")
        for (s, t), h in list(zip(bleu_pairs, hyps))[:3]:
            print(f"   fr : {' '.join(s)}\n   ref: {' '.join(t)}\n   hyp: {' '.join(h)}")

        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos},
                       os.path.join(cfg.out_dir, "best.pt"))
            print(f"   保存 best.pt (valid loss {best:.3f})")

    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print(f"完成。评估: python evaluate.py --out_dir {cfg.out_dir}")


if __name__ == "__main__":
    train(Config.from_args()[0])
