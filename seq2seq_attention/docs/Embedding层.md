# Embedding 层: 用的是 word2vec 吗?

不是。Seq2Seq 里的 embedding 层是 PyTorch 的 `nn.Embedding`, **随机初始化, 然后跟整个模型一起训练**, 没有用 word2vec 的向量。

## 1. 代码里是怎么做的

[model.py](../model.py) 里有两个 embedding:

```python
self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)   # Encoder: 法语, 13,307 × 256
self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)   # Decoder: 英语,  8,479 × 256
```

- `nn.Embedding` 本质上就是一张 $[|V| \times E]$ 的查找表。输入词 id, 取出对应的那一行。
- 初始化是 $\mathcal{N}(0, 0.01^2)$ 的随机数, `<pad>` 那一行置 0 (见 `Seq2Seq._init_weights`)。
- 训练时它和 GRU、attention 用**同一个 loss、同一个优化器**一起更新, 梯度来自翻译的交叉熵。学出来的向量是"对翻译有用"的表示, 不是 word2vec 那种"上下文相似"的表示。

Bahdanau 的论文也是这样做的: embedding 随机初始化, 跟模型一起训练。

## 2. 和 word2vec 是什么关系

word2vec 里的 `in_embed` 和这里的 `nn.Embedding` 是**同一种东西**, 都是查找表, 区别只在于怎么训练:

| | word2vec | Seq2Seq 里的 embedding |
|---|---|---|
| 结构 | 查找表 $[|V| \times D]$ | 查找表 $[|V| \times E]$ |
| 训练目标 | 预测上下文词 (负采样) | 翻译的交叉熵 |
| 训练方式 | 单独训练, 用完拿去别处 | 和整个模型端到端一起训练 |
| 学到的是 | 通用的词语义 | 专门为这个翻译任务服务的表示 |

## 3. 能不能用训练好的 word2vec 来初始化

技术上可以, 这叫"用预训练词向量初始化": 先把 word2vec 的向量拷进 `nn.Embedding`, 再继续训练或者冻结。但放在这个项目里不太合适:

1. **语言对不上**: 本仓库的 word2vec 是在 text8 上训练的, 只有英文。编码器的输入是法语, 用不上; 只有解码器的英文 embedding 能用。
2. **维度对不上**: word2vec 是 100 维, 这里是 256 维, 要么改 `emb_dim`, 要么加一层映射。
3. **词表对不上**: text8 没有标点, 也没有 `i'` 这种按撇号拆出来的 token, 这些词只能随机初始化。
4. **收益不大**: 训练集有 11.8 万句对, 数据足够让 embedding 从零学好。预训练词向量主要在数据很少时有帮助。

预训练的思路后来被推得更远: 到了 BERT 和 GPT, 不只是 embedding 这一层, 而是整个模型都先预训练再微调 (路线图第 4、5 步)。
