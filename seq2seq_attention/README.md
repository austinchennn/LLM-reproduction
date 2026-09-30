# Seq2Seq + Attention 手搓复现

论文: *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau, Cho, Bengio 2014)

前置: *Sequence to Sequence Learning with Neural Networks* (Sutskever 2014)、*Learning Phrase Representations using RNN Encoder-Decoder* (Cho 2014)

## 要解决的问题

普通 Seq2Seq 把整句源语言压成**一个固定长度向量**, 再让解码器从这个向量生成译文。句子一长, 这个向量装不下所有信息, 翻译质量明显下降。

Bahdanau 的做法: 编码器保留**每个词位置的隐藏状态**, 解码器每生成一个词, 都重新计算"现在该看源句的哪几个词", 按权重加权求和得到一个**动态的上下文向量**。这就是 attention 的起源, 后来的 Transformer 把它推广成了 self-attention。

## 代码结构

| 文件 | 内容 |
|---|---|
| `config.py` | `Config` dataclass: 全部超参, 命令行 `--xxx` 覆盖同名字段 (与 word2vec 一致) |
| `data.py` | 下载/读取英法句对、清洗、分词、建词表、按长度过滤、padding + batch |
| `model.py` | `Encoder` (双向 GRU)、`BahdanauAttention` (加性注意力)、`Decoder` (单向 GRU)、`Seq2Seq` |
| `translate.py` | 推理: greedy / beam search 解码, 画注意力热力图 |
| `evaluate.py` | 困惑度、BLEU (自己实现的 corpus BLEU-4)、按源句长度分组的 BLEU |
| `train.py` | 训练入口: teacher forcing、梯度裁剪、验证、保存 checkpoint |

原理笔记在 [docs/](docs/): [GRU.md](docs/GRU.md) (门控公式、与 LSTM 对比)、[损失函数.md](docs/损失函数.md) (句子的交叉熵与困惑度)、[BLEU.md](docs/BLEU.md) (评价指标的定义、公式与计算例子)。

## 运行

```bash
python train.py                                             # 默认数据 data/eng-fra.txt, 不存在自动下载
python train.py --max_pairs 5000 --epochs 2                 # 快速调试
python train.py --no_attention true --out_dir output_noattn # 无 attention 的 baseline
python evaluate.py                                          # test 集: 困惑度、BLEU、按长度分组 BLEU
python translate.py "elle est plus grande que moi ." --plot  # 翻译一句并画注意力热力图
python translate.py                                         # 交互模式
```

输出在 `output/`: `best.pt` (valid loss 最好的模型, 含词表和超参)、`test.tsv` (test 集句对)、`config.json`、`attention_*.png`。
所有 `Config` 字段都能用 `--xxx` 覆盖, 布尔值写 `true/false`。

## 数据

- **Tatoeba 英法句对**, 用 PyTorch 官方教程的镜像 (<https://download.pytorch.org/tutorial/data.zip> 中的 `eng-fra.txt`), 约 13.6 万对, 每行 `英文 \t 法文`。manythings.org 原站会拦截脚本下载, 所以不用它。
- 方向: **法 → 英** (fr → en)。英文作目标语言, 方便人直接看输出好坏。
- 预处理:
  1. Unicode 规范化 (NFD) 后去掉重音符号, 全部小写 (`Ça va?` → `ca va ?`)。
  2. 标点前后加空格, 撇号后加空格 (`j'ai` → `j' ai`), 按空格分词 (词级别, 不用 BPE —— BPE 留到 Transformer 那一步)。
  3. 只保留源句和目标句都 **≤ 15 个词** 的句对 (过滤后约 13 万对)。Mac MPS 上每个 epoch 约 2.5 分钟, 默认 15 个 epoch 约 40 分钟; 第 1 个 epoch 结束 BLEU 就能到 30 左右。
  4. 去掉完全重复的句对。
  5. `min_count = 2`: 训练集中出现少于 2 次的词记为 `<unk>`。
  6. 随机划分 train / valid / test = 90% / 5% / 5%。
- 特殊符号: `<pad>=0`、`<unk>=1`、`<sos>=2`、`<eos>=3`。

## 模型结构

### 1. 编码器 (Encoder): 双向 GRU

```
源句 x_1 ... x_T  (词 id)
  → Embedding (V_src × E, E=256)
  → 双向 GRU (hidden H=512)
      前向: →h_j  读 x_1 → x_T
      后向: ←h_j  读 x_T → x_1
  → h_j = [→h_j ; ←h_j]        每个位置 2H = 1024 维, 叫 "annotation"
```

- **为什么双向**: 每个 h_j 既包含 x_j 左边的信息, 也包含右边的信息, 代表"以 x_j 为中心的整句摘要"。
- **为什么 GRU 不用 LSTM**: 论文用的就是 GRU (Cho 2014 提出的门控单元)。参数比 LSTM 少, 效果接近。
- 输出两样东西:
  - 所有位置的 `annotations`: `[B, T, 2H]`, 给 attention 用。
  - 解码器初始状态: `s_0 = tanh(W_s · ←h_1)` (论文附录 A.2.2, 用后向 GRU 读完整句后的最后状态), 形状 `[B, H]`。

### 2. 注意力 (Bahdanau / 加性注意力)

解码第 i 步时, 用上一步的解码器状态 s_{i-1} 去"查询"所有 annotations:

```
e_ij  = v_a^T · tanh(W_a · s_{i-1} + U_a · h_j)       对齐分数 (alignment score), 标量
α_ij  = softmax_j(e_ij)                                在源句位置 j 上归一化, 并 mask 掉 <pad>
c_i   = Σ_j α_ij · h_j                                  上下文向量, [B, 2H]
```

- 叫"加性"是因为 query 和 key 先各自线性变换再**相加**, 过 tanh, 而不是直接点积 (点积版是 Luong 2015, 以及 Transformer 的 scaled dot-product)。
- `U_a · h_j` 与解码步无关, **在整句上只算一次**缓存下来, 每步只算 `W_a · s_{i-1}`。
- attention 维度 A = 512。

### 3. 解码器 (Decoder): 单向 GRU + attention

每一步:

```
输入: 上一个词 y_{i-1}、上一步状态 s_{i-1}、全部 annotations
1. c_i = Attention(s_{i-1}, annotations)
2. s_i = GRU( [Embed(y_{i-1}) ; c_i],  s_{i-1} )
3. t_i = Maxout( W_o · [s_i ; Embed(y_{i-1}) ; c_i] )     论文用 maxout (pool 大小 2), 可改成 tanh 简化
4. p(y_i | y_<i, x) = softmax( W_out · t_i )              对整个目标词表 V_tgt
```

注意 attention 用的是 **s_{i-1}** (先看再写), 这是 Bahdanau 与 Luong (用 s_i, 先写再看) 的主要区别之一。

### 超参 (默认)

| 参数 | 值 | 论文原值 |
|---|---|---|
| embedding E | 256 | 620 |
| GRU hidden H | 512 | 1000 |
| attention 维 A | 512 | 1000 |
| maxout 输出 | 256 | 500 |
| 最大句长 | 15 | 30 / 50 |
| batch | 64 | 80 |
| 优化器 | Adam, lr 1e-3 | Adadelta (ε=1e-6, ρ=0.95) |
| 梯度裁剪 | L2 范数 1.0 | 1.0 |
| dropout | 0.3 (embedding 与输出层) | 无 |
| epochs | 10~15 | ~5 天 |

## 训练过程

### 目标函数

最大化目标句的对数似然, 即对每个目标位置做**完整 softmax 的交叉熵**, 忽略 `<pad>`:

```
loss = - (1/N) Σ_i log p(y_i | y_<i, x)           N 为该 batch 中非 pad 的目标词数
```

### Teacher Forcing

训练时解码器第 i 步的输入 **用真实的 y_{i-1}**, 而不是模型自己上一步预测的词。这样所有步的输入都已知, 训练稳定、收敛快。
可选 `--teacher_forcing 0.9`: 以 10% 概率改用模型自己的预测, 缓解训练/推理不一致 (exposure bias)。默认 1.0, 与论文一致。

### 例子: 一个句对走一遍

句对: `je suis etudiant .` → `i am a student .`

```
源 ids:   [je, suis, etudiant, ., <eos>]                         T = 5
解码输入: [<sos>, i,  am, a,  student, .      ]                  目标左移一位
解码目标: [i,     am, a,  student, .,  <eos>  ]
```

1. **编码**: 5 个词 → Embedding `[1,5,256]` → 双向 GRU → annotations `[1,5,1024]`; s_0 由后向 GRU 在 `je` 处的状态得到 `[1,512]`。
2. **解码第 1 步**: 输入 `<sos>` 与 s_0。
   - attention 算出 α_1 ≈ `[0.70, 0.20, 0.05, 0.03, 0.02]` → 主要看 `je`
   - c_1 = Σ α_1j h_j, 与 Embed(`<sos>`) 拼接进 GRU 得 s_1
   - 输出 softmax, 取目标 `i` 的概率, 假设 p=0.40 → loss_1 = -log 0.40 = 0.92
3. **第 2 步**: 输入 **真实的** `i` (teacher forcing), attention 主要看 `suis` → 目标 `am`。
4. **第 3 步**: 目标 `a` —— 法语里 `etudiant` 前没有冠词, attention 会分散在 `suis` / `etudiant` 上。这正是"软对齐"比硬对齐好的地方。
5. ... 直到第 6 步目标 `<eos>`。
6. 6 步 loss 平均 → 反向传播 → 梯度裁剪到范数 1 → Adam 更新。

实际实现中 batch 内按 pad 对齐一起算, 上面只是单样本的展开。

### 训练循环中的细节

- **按长度排序分桶**: 同一 batch 句长相近, 减少 padding 浪费。
- **`pack_padded_sequence`**: 编码器 GRU 跳过 pad, 后向 GRU 才能从真正的最后一个词开始读。
- 每个 epoch 结束: 在 valid 上计算 loss / **困惑度 (perplexity = exp(loss))**, 用 greedy 解码算 BLEU, 打印几条样例翻译; 保存 valid loss 最好的 checkpoint。
- **学习率调度**: `ReduceLROnPlateau`, valid loss 两个 epoch 不降就减半。

## 推理过程

推理时没有目标句, 解码器只能**把自己上一步输出的词喂回去** (自回归), 直到输出 `<eos>` 或达到最大长度 (源句长 × 2)。

### Greedy 解码

每一步直接取概率最大的词。快, 但一步错会步步错。

### Beam Search (默认, beam = 5, 论文做法)

每一步保留累计 log 概率最高的 k 条候选, 每条展开 V 个词, 再从 k×V 中选出前 k 条; 候选输出 `<eos>` 就移入完成列表。最后按**长度归一化分数** `log p / len^α` (α=0.7) 选最好的一条, 否则 beam search 会偏好短句。

### 例子: `elle est plus grande que moi .`

```
编码: 7 个 token → annotations [1,7,1024], s_0

步 1  输入 <sos>          attention 峰值: elle      beam: she(-0.1)  it(-2.6) ...
步 2  输入 she            attention 峰值: est       beam: she is(-0.3)  she's(-1.9) ...
步 3  输入 is             attention 峰值: grande    beam: she is taller(-0.9)  she is bigger(-1.2) ...
步 4  输入 taller         attention 峰值: que       → than
步 5  输入 than           attention 峰值: moi       → me
步 6  输入 me             attention 峰值: .         → .
步 7  输入 .                                         → <eos>, 结束

输出: she is taller than me .
```

第 3 步很能说明问题: 法语 `plus grande` (更 + 高) 两个词对应英文一个词 `taller`, attention 会同时落在 `plus` 和 `grande` 上。

### 注意力可视化

`translate.py --plot` 把每一步的 α_i 画成 `[目标长度 × 源长度]` 热力图 (论文 Figure 3)。对于词序一致的句子, 热力图接近对角线; 对于 `la zone economique europeenne` → `the european economic area` 这种形容词顺序颠倒的, 能看到反对角的块。这是检验 attention 是否学对的最直观手段。

## 用到的技术细节 (与 word2vec 对比)

| 技术 | 是否使用 | 说明 |
|---|---|---|
| **负采样** | ❌ 不用 | 目标词表只有 ~1 万, 完整 softmax 每步只需 `[B,256]×[256,V]`, 在 GPU / MPS 上很便宜。负采样本质是训练词向量的近似目标, 不能直接给出归一化概率, 推理和 beam search 都需要真实概率。论文也是完整 softmax (3 万词表)。词表很大时的替代方案是 sampled softmax (Jean 2015), 本项目不需要。 |
| **稀疏梯度 / SparseAdam** | ❌ 不用 | word2vec 的参数几乎全是 embedding, 每步只触及少数行, 稀疏化收益大。这里 GRU、attention、输出层都是**稠密**参数, 每步全部更新; embedding 只占一部分, 且词表小。用普通 `Adam` 更简单。 |
| **下采样高频词** | ❌ 不用 | 翻译要保留完整句子, 丢掉 "the"、"de" 会改变句意。 |
| **完整 softmax + 交叉熵** | ✅ | `CrossEntropyLoss(ignore_index=PAD)`, 内部是 `log_softmax`, 数值稳定。 |
| **padding + mask** | ✅ | 三处: 编码器 `pack_padded_sequence`; attention 分数在 pad 位置置 `-inf` 再 softmax; loss 忽略 pad。 |
| **Teacher forcing** | ✅ | 见上文。 |
| **梯度裁剪** | ✅ | `clip_grad_norm_(1.0)`, RNN 防梯度爆炸的标配。 |
| **Dropout** | ✅ | 论文没有, 但小数据不加很容易过拟合。 |
| **权重初始化** | ✅ | 循环权重正交初始化 (与论文一致), 其他 `N(0, 0.01²)`, bias 为 0。 |
| **Beam search + 长度归一化** | ✅ | 推理用。 |
| **权重共享 (目标 embedding 与输出层)** | 可选 `--tie_weights true` | 论文没有, 需要 maxout 输出维 = embedding 维, 能减少参数。 |
| **BPE / 子词** | ❌ | 词级别足够, 子词留到 Transformer。 |

## 评估

- **困惑度** (valid / test)。
- **BLEU** (test 集, beam=5)。`evaluate.corpus_bleu` 自己实现的 corpus BLEU-4 (截断 n-gram 精确率的几何平均 × 简短惩罚), 不依赖 sacrebleu; 分词是本项目自己的, 所以数值不能与论文或其他项目直接比较。
- **对比实验** (论文 Figure 2 的精简版): 关掉 attention (`--no_attention true`, 解码器每步都看同一个固定向量 [→h_T ; ←h_1]) 训一个 baseline, 按源句长度分组比较 BLEU, 应能看到句子越长, 有 attention 的优势越大。

## 未实现的细节

- 不做论文中的 30k 高频词表 + 大规模 WMT'14 语料, 只用 Tatoeba 短句。
- 不做 coverage / 输入 feeding (Luong 2015) 等后续改进。
- 不做 `<unk>` 替换 (按 attention 从源句复制词)。
