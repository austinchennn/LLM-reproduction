"""评估: 余弦相似度近邻 & 类比"""

import torch.nn.functional as F


class Evaluator:
    def __init__(self, model, vocab):
        self.vocab = vocab
        self.emb = F.normalize(model.embeddings().float().cpu(), dim=1)

    def most_similar(self, word, topk=8):
        if word not in self.vocab.stoi:
            return []
        sims = self.emb @ self.emb[self.vocab.stoi[word]]
        idx = sims.topk(topk + 1).indices.tolist()
        return [(self.vocab.itos[i], round(sims[i].item(), 3)) for i in idx if self.vocab.itos[i] != word][:topk]

    def analogy(self, a, b, c, topk=5):
        """a - b + c ≈ ?  例: king - man + woman ≈ queen"""
        if any(w not in self.vocab.stoi for w in (a, b, c)):
            return []
        s = self.vocab.stoi
        q = F.normalize(self.emb[s[a]] - self.emb[s[b]] + self.emb[s[c]], dim=0)
        sims = self.emb @ q
        sims[[s[a], s[b], s[c]]] = -1
        idx = sims.topk(topk).indices.tolist()
        return [(self.vocab.itos[i], round(sims[i].item(), 3)) for i in idx]


def report(model, vocab, probes, analogies):
    ev = Evaluator(model, vocab)
    for w in probes:
        res = ev.most_similar(w)
        if res:
            print(f"  {w:>10} -> {', '.join(x for x, _ in res)}")
    for a, b, c in analogies:
        res = ev.analogy(a, b, c)
        if res:
            print(f"  {a} - {b} + {c} -> {', '.join(x for x, _ in res)}")
