"""数据: 下载英法句对、清洗分词、建词表、划分、按长度分桶生成 batch"""

import io
import os
import random
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter

import torch

# Tatoeba 英法句对, PyTorch 官方教程的镜像 (manythings.org 会拦截非浏览器下载)
URL = "https://download.pytorch.org/tutorial/data.zip"

PAD, UNK, SOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]


def download(path):
    print(f"下载 {URL} ...")
    with urllib.request.urlopen(URL) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(zf.read("data/eng-fra.txt"))


def normalize(s):
    """去重音、小写、标点和撇号拆开: "J'ai froid !" -> "j' ai froid !" """
    s = s.replace(" ", " ").replace("\xa0", " ")           # 法语标点前的不换行空格
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # 去掉重音符号
    s = s.lower().replace("’", "'")
    s = re.sub(r"([.!?,;:\"()])", r" \1 ", s)
    s = re.sub(r"'", "' ", s)
    s = re.sub(r"[^a-z0-9.!?,;:\"()'\- ]+", " ", s)
    return s.split()


def load_pairs(path, max_len, max_pairs=None):
    """返回 [(法文 tokens, 英文 tokens)], 方向 fr -> en; 去掉过长句和重复句对"""
    if not os.path.exists(path):
        download(path)
    pairs, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            en, fr = normalize(cols[0]), normalize(cols[1])
            if not (0 < len(fr) <= max_len and 0 < len(en) <= max_len):
                continue
            key = (" ".join(fr), " ".join(en))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((fr, en))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def split_pairs(pairs, valid_ratio, test_ratio, seed):
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    n_valid, n_test = int(len(pairs) * valid_ratio), int(len(pairs) * test_ratio)
    return pairs[n_valid + n_test:], pairs[:n_valid], pairs[n_valid:n_valid + n_test]


class Vocab:
    def __init__(self, sentences=None, min_count=1, itos=None):
        if itos is None:
            counts = Counter(w for s in sentences for w in s)
            itos = SPECIALS + sorted(w for w, c in counts.items() if c >= min_count)
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(w, UNK) for w in tokens]

    def decode(self, ids):
        """遇到 <eos> 停止, 跳过 <sos>/<pad>"""
        out = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, SOS):
                out.append(self.itos[i])
        return out


def encode_pairs(pairs, src_vocab, tgt_vocab):
    """源句末尾加 <eos>; 目标句保留原始 id, 拼 <sos>/<eos> 在 collate 里做"""
    return [(src_vocab.encode(s) + [EOS], tgt_vocab.encode(t)) for s, t in pairs]


def collate(batch):
    """batch: [(src_ids, tgt_ids)] -> src [B,T], src_len [B], tgt_in [B,m], tgt_out [B,m]

    tgt_in  = <sos> y_1 ... y_n      (teacher forcing 的输入, 目标左移一位)
    tgt_out = y_1 ... y_n <eos>      (每一步要预测的词)
    """
    B = len(batch)
    T = max(len(s) for s, _ in batch)
    m = max(len(t) for _, t in batch) + 1
    src = torch.full((B, T), PAD, dtype=torch.long)
    tgt_in = torch.full((B, m), PAD, dtype=torch.long)
    tgt_out = torch.full((B, m), PAD, dtype=torch.long)
    src_len = torch.tensor([len(s) for s, _ in batch], dtype=torch.long)
    for b, (s, t) in enumerate(batch):
        src[b, :len(s)] = torch.tensor(s)
        tgt_in[b, :len(t) + 1] = torch.tensor([SOS] + t)
        tgt_out[b, :len(t) + 1] = torch.tensor(t + [EOS])
    return src, src_len, tgt_in, tgt_out


def iter_batches(data, batch_size, shuffle=True, rng=None):
    """按源句长度分桶: 每 100 个 batch 的数据为一块, 块内按长度排序后切 batch, 再打乱 batch 顺序"""
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * 100
    batches = []
    for i in range(0, len(idx), chunk):
        block = sorted(idx[i:i + chunk], key=lambda j: len(data[j][0]))
        batches += [block[k:k + batch_size] for k in range(0, len(block), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    for bidx in batches:
        yield collate([data[j] for j in bidx])
