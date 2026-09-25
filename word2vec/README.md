# Word2Vec 手搓复现

论文: *Efficient Estimation of Word Representations in Vector Space* (Mikolov 2013a),
*Distributed Representations of Words and Phrases and their Compositionality* (Mikolov 2013b)

## 代码结构

| 文件 | 内容 |
|---|---|
| `config.py` | `Config` dataclass: 全部超参及默认值, 命令行 `--xxx` 会自动覆盖同名字段 |
| `data.py` | 读语料、建词表、高频词下采样、负采样噪声表、分块生成训练 batch |
| `model.py` | `Word2Vec`: 两套 embedding + 负采样损失, `forward_sg` / `forward_cbow` |
| `evaluate.py` | 余弦近邻、类比 |
| `train.py` | 训练入口: 组装数据/模型/优化器, 训练循环, 保存结果 |

## 运行

```bash
python train.py --mode sg                   # Skip-gram, 默认自动下载 text8 (~100MB, 1700 万词)
python train.py --mode cbow
python train.py --mode sg --max_tokens 2000000 --epochs 1   # 快速调试
python train.py --corpus my.txt             # 自己的语料: 空格分词的纯文本, 中文先用 jieba 分词
```

输出在 `output/<mode>_d<dim>/`: `embeddings.npy`、`vectors.txt` (word2vec 文本格式, gensim 可直接加载) 和 `config.json` (本次使用的超参)。

## 两种结构

| | Skip-gram | CBOW |
|---|---|---|
| 输入 → 预测 | 中心词 → 每个上下文词 | 上下文词向量平均 → 中心词 |
| 每个中心词产生的样本数 | 约 2b 个 (b 为窗口) | 1 个 |
| 速度 | 慢 (样本多) | 快数倍 |
| 效果 | 低频词、语义类比更好 | 高频词、语法类比稍好, 更平滑 |

**工程上怎么选**: 从目前纯工程化落地的角度看, 绝对的主流是 **Skip-gram**, 而且几乎必然搭配负采样 (Negative Sampling), 即 **SGNS** (Skip-Gram with Negative Sampling)。例如 fastText 的 `skipgram` 默认就是负采样; gensim 虽然默认 `sg=0` (CBOW), 实际使用时通常会设 `sg=1, negative=5~20`; 推荐系统里的 Item2Vec、图嵌入里的 Node2Vec 用的也是 SGNS。CBOW 和 hierarchical softmax 现在主要出现在教学和对比实验里。

## 负采样

full softmax 每步要对 |V| 个词 (text8 约 7 万) 做点积并归一化。负采样把它变成 1 个正样本 + K 个噪声词的二分类:

```
loss = -log σ(u_o · v) - Σ_k log σ(-u_k · v),   u_k ~ P_n(w) ∝ count(w)^0.75
```

每个正样本的计算量从 O(|V|·D) 降到 O((K+1)·D)。

## 用到的工程细节

1. **min_count 过滤低频词**: 出现不到 5 次的词直接丢掉, 这些词学不出好向量, 还会让词表变大。
2. **高频词下采样** (`--subsample 1e-4`): 按 `keep = (sqrt(f/t)+1)·t/f` 随机丢弃 "the"、"of" 等高频词, 每个 epoch 重新采样。这样训练更快, 低频词的向量也更好。
3. **噪声分布取 0.75 次方 + 预展开采样表**: 表大小为 1e7, 采样时只需 `randint` 取下标, 比 `multinomial` 快。
4. **动态窗口**: 每个中心词的实际窗口 b ~ U[1, W], 相当于让近处的上下文权重更高。
5. **两套 embedding (in / out)**: 输入侧均匀初始化到 ±0.5/D, 输出侧初始化为全 0 (与原版 C 代码一致)。最终使用 in_embed。
6. **稀疏梯度** (`sparse=True` + `SparseAdam`): 每步只更新 batch 中出现的行, 词表越大收益越明显。
7. **学习率线性衰减**到初始值的 1e-4 倍 (原版做法), 总步数按下采样后的期望词数估计。
8. **CBOW 的 padding + mask 平均**: 窗口不满时补 pad, pad 不参与平均, 也不接收梯度。
9. **分块生成样本并在块内打乱**: text8 用 Skip-gram 会产生约 5000 万个 pair, 不一次性放进内存。
10. **`logsigmoid`** 代替 `log(sigmoid(x))`, 数值更稳定。
11. **评估**: 余弦近邻 + 类比 (king - man + woman ≈ queen), 每个 epoch 打印一次。

未实现的细节: 负样本碰到正样本时不剔除 (原版也不处理)；不做短语合并 (如 "new_york", 论文 2013b 第 4 节)；不实现 hierarchical softmax (负采样的替代方案)。

## Embedding 维度一般取多少

- **常用 100 ~ 300**。Google News 预训练向量是 300 维, GloVe 提供 50/100/200/300 维。
- 语料小 (百万到千万词, 如 text8): **50 ~ 100** 就够, 维度再高容易欠训练。
- 语料大 (十亿词以上): **300** 左右, 再往上收益很小。
- 经验: 维度应随语料规模和词表大小增长。维度太小会欠拟合, 语义挤在一起; 维度太大则训练慢, 需要更多数据才能训好。
- 对照: Transformer 的 d_model (GPT-2 small 为 768) 比 word2vec 大得多, 因为它的 embedding 只是深层网络的输入, 表达能力主要来自后面的网络层。

## 其他超参经验值

| 参数 | Skip-gram | CBOW |
|---|---|---|
| window | 5 ~ 10 | 5 |
| neg (K) | 5 ~ 20 (小语料), 2 ~ 5 (大语料) | 同左 |
| subsample t | 1e-3 ~ 1e-5 | 同左 |
| epochs | 3 ~ 5 (text8) | 同左 |
