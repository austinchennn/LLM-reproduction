"""推理: greedy / beam search 解码, 注意力热力图

用法:
    python translate.py "je suis etudiant ."
    python translate.py "elle est plus grande que moi ." --plot          # 保存注意力热力图
    python translate.py "il fait froid ." --beam_size 1                 # greedy
    python translate.py                                                  # 交互模式, 逐行输入法语
"""

import argparse
import os

import torch
import torch.nn.functional as F

from config import Config
from data import EOS, SOS, Vocab, normalize
from model import Seq2Seq


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**{**ckpt["config"], "device": str(device)})
    src_vocab, tgt_vocab = Vocab(itos=ckpt["src_itos"]), Vocab(itos=ckpt["tgt_itos"])
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, src_vocab, tgt_vocab, cfg


@torch.no_grad()
def greedy_decode(model, src, src_len, max_len):
    """batch 版 greedy: 每步取概率最大的词喂回去。返回 ids [B, ≤max_len]"""
    enc, s = model.encode(src, src_len)
    y = torch.full((src.size(0),), SOS, dtype=torch.long, device=src.device)
    out, done = [], torch.zeros_like(y, dtype=torch.bool)
    for _ in range(max_len):
        logits, s, _ = model.decoder.step(y, s, enc)
        y = logits.argmax(-1)
        out.append(y)
        done |= y == EOS
        if done.all():
            break
    return torch.stack(out, dim=1)


def _expand(enc, n):
    """把 batch=1 的编码结果复制成 n 份 (expand 不拷贝内存)"""
    return {k: (v.expand(n, *v.shape[1:]) if v is not None else None) for k, v in enc.items()}


@torch.no_grad()
def beam_search(model, src, src_len, beam_size, len_alpha, max_len):
    """单句 beam search。src [1,T]。返回 (ids 列表, 注意力 [len, T] 或 None)

    每一步: 每条候选展开 V 个词, 累计 log 概率, 从 n×V 里取前 k; 以 <eos> 结尾的候选移入完成列表,
    beam 相应缩小。最后按 log p / len^alpha 选最好的一条, 避免偏好短句。
    """
    enc1, s = model.encode(src, src_len)
    device = src.device
    tokens = torch.full((1, 1), SOS, dtype=torch.long, device=device)  # 含开头的 <sos>
    scores = torch.zeros(1, device=device)
    attn_hist = torch.zeros(1, 0, src.size(1), device=device)
    finished = []  # (归一化分数, ids, 注意力)

    for t in range(max_len):
        n = tokens.size(0)
        logits, s_new, alpha = model.decoder.step(tokens[:, -1], s, _expand(enc1, n))
        total = scores.unsqueeze(1) + F.log_softmax(logits, dim=-1)       # [n,V]
        k = beam_size - len(finished)
        top_scores, top_idx = total.view(-1).topk(k)
        V = logits.size(-1)
        prev, word = top_idx // V, top_idx % V

        keep = []
        for j in range(k):
            p, w, sc = prev[j].item(), word[j].item(), top_scores[j].item()
            if w == EOS or t == max_len - 1:
                ids = tokens[p, 1:].tolist() + ([w] if w != EOS else [])
                att = torch.cat([attn_hist[p], alpha[p:p + 1]], 0) if alpha is not None else None
                finished.append((sc / (len(ids) + 1) ** len_alpha, ids, att))
            else:
                keep.append(j)
        if not keep or len(finished) >= beam_size:
            break
        keep = torch.tensor(keep, device=device)
        p_idx = prev[keep]
        tokens = torch.cat([tokens[p_idx], word[keep].unsqueeze(1)], dim=1)
        scores = top_scores[keep]
        s = s_new[p_idx]
        if alpha is not None:
            attn_hist = torch.cat([attn_hist[p_idx], alpha[p_idx].unsqueeze(1)], dim=1)

    _, ids, att = max(finished, key=lambda x: x[0])
    return ids, att


def translate(model, src_vocab, tgt_vocab, text, device, beam_size=5, len_alpha=0.7):
    """法语原文 -> (源 tokens, 英文 tokens, 注意力 [len_tgt(含 <eos>), len_src(含 <eos>)])"""
    src_tokens = normalize(text)
    ids = src_vocab.encode(src_tokens) + [EOS]
    src = torch.tensor([ids], device=device)
    src_len = torch.tensor([len(ids)])
    max_len = 2 * len(ids) + 2
    if beam_size <= 1:
        out = greedy_decode(model, src, src_len, max_len)[0].tolist()
        return src_tokens, tgt_vocab.decode(out), None
    out, att = beam_search(model, src, src_len, beam_size, len_alpha, max_len)
    src_shown = [w if w in src_vocab.stoi else f"{w}(unk)" for w in src_tokens]
    return src_shown, tgt_vocab.decode(out), att


def plot_attention(src_tokens, tgt_tokens, attn, path):
    """画 [目标长度 × 源长度] 的注意力热力图 (论文 Figure 3)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src_labels, tgt_labels = src_tokens + ["<eos>"], tgt_tokens + ["<eos>"]
    a = attn[:len(tgt_labels), :len(src_labels)].cpu().numpy()
    fig, ax = plt.subplots(figsize=(0.6 * len(src_labels) + 2, 0.6 * len(tgt_labels) + 1.5))
    ax.imshow(a, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks(range(len(src_labels)), src_labels, rotation=60, ha="left")
    ax.set_yticks(range(len(tgt_labels)), tgt_labels)
    ax.xaxis.tick_top()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default=None, help="法语句子; 不给则进入交互模式")
    p.add_argument("--ckpt", default=None, help="默认 <out_dir>/best.pt")
    p.add_argument("--plot", action="store_true", help="保存注意力热力图到 <out_dir>/attention_*.png")
    cfg, ns = Config.from_args(parser=p)
    device = torch.device(cfg.device)
    model, src_vocab, tgt_vocab, train_cfg = load_checkpoint(ns.ckpt or os.path.join(cfg.out_dir, "best.pt"),
                                                             device)
    if ns.plot and train_cfg.no_attention:
        print("该模型没有 attention, 无法画热力图")
        ns.plot = False

    def run(text):
        src, out, att = translate(model, src_vocab, tgt_vocab, text, device, cfg.beam_size, cfg.len_alpha)
        print(f"> {' '.join(src)}\n= {' '.join(out)}")
        if ns.plot and att is not None:
            name = "_".join(normalize(text))[:40].replace("/", "")
            path = os.path.join(cfg.out_dir, f"attention_{name}.png")
            plot_attention(src, out, att, path)
            print(f"热力图: {path}")

    if ns.text:
        run(ns.text)
    else:
        try:
            while True:
                line = input("fr> ").strip()
                if line:
                    run(line)
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
