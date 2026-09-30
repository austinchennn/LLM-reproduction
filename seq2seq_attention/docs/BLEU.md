# BLEU (Bilingual Evaluation Understudy)

论文: *BLEU: a Method for Automatic Evaluation of Machine Translation* (Papineni et al. 2002)

在本项目中的位置: `evaluate.corpus_bleu`, 训练时每个 epoch 在 valid 上算一次 (greedy), 测试时在 test 上算 (beam=5)。

## 1. 定义: 它只是评价指标

BLEU 衡量**模型译文 (candidate) 与人工参考译文 (reference) 的 n-gram 重合程度**, 取值 0~1, 习惯上乘 100 报告, 越高越好。

它**不是 loss, 也不是正则项, 不参与训练**:

| | 训练用的 loss (交叉熵, 见 [损失函数.md](损失函数.md)) | 正则项 (如 $\lambda \lVert W \rVert^2$、dropout) | BLEU |
|---|---|---|---|
| 位置 | 训练目标本身 | 加进 loss 或作用于网络 | 训练之外单独计算 |
| 作用 | 让模型给正确词高概率 | 约束模型, 防止过拟合 | 衡量解码出的整句译文好不好 |
| 可导 | 是 | 是 | **否** |

**为什么不能直接拿来优化 (梯度下降)**:

1. **不可导**: 算 BLEU 之前必须先解码出离散的词 (argmax / beam search), 再数 n-gram 个数。"取 argmax" 和 "计数" 都没有梯度, 反向传播传不回模型参数。
2. **句子级不稳定**: BLEU 是为整个语料设计的; 单句上某个 $p_n$ 很容易为 0, 整句 BLEU 直接变 0, 信号极不平滑。

所以本项目的流程是: **用交叉熵训练, 用 BLEU 评估** (以及可选地用来挑 checkpoint)。

> 补充: 确实有工作把 BLEU 当作 **reward**, 用强化学习 (策略梯度, 如 MIXER, Ranzato et al. 2015) 或最小风险训练 (MRT, Shen et al. 2016) 间接优化它。这是绕开不可导的特殊技巧, 本项目不涉及。

## 2. 公式

### 2.1 截断 n-gram 精确率 $p_n$

对 $n = 1, 2, 3, 4$, 统计候选译文的 n-gram 里有多少出现在参考译文中。每个 n-gram 的命中数**截断 (clip)** 到它在参考里出现的次数:

$$
p_n = \frac{\sum_{g \in \text{n-grams(cand)}} \min\big(\mathrm{Count}_{cand}(g),\ \mathrm{Count}_{ref}(g)\big)}{\sum_{g \in \text{n-grams(cand)}} \mathrm{Count}_{cand}(g)}
$$

有多个参考译文时, $\mathrm{Count}_{ref}(g)$ 取各参考中的最大值。

### 2.2 简短惩罚 BP (Brevity Penalty)

$p_n$ 只看精确率, 输出越短越容易高分, 所以要惩罚过短的译文。设候选总长 $c$, 参考总长 $r$:

$$
BP = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}
$$

### 2.3 合成

$$
\mathrm{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right), \qquad N = 4,\ w_n = \tfrac{1}{4}
$$

即 4 个精确率的**几何平均**再乘 BP, 这就是常说的 BLEU-4。

## 3. 计算例子

```
参考 (r = 8): the cat is sitting on the mat .
候选 (c = 7): the cat is on the mat .
```

| n | 候选中的 n-gram | 命中 | $p_n$ | $\log p_n$ |
|---|---|---|---|---|
| 1 | the, cat, is, on, the, mat, . | 全部 7 个 (`the` 在参考里出现 2 次, 不被截断) | 7/7 = 1.000 | 0.000 |
| 2 | the cat, cat is, **is on**, on the, the mat, mat . | 除 `is on` 外 5 个 | 5/6 = 0.833 | -0.182 |
| 3 | the cat is, **cat is on**, **is on the**, on the mat, the mat . | 3 个 | 3/5 = 0.600 | -0.511 |
| 4 | **the cat is on**, **cat is on the**, **is on the mat**, on the mat . | 1 个 | 1/4 = 0.250 | -1.386 |

(加粗的是参考里没有的 n-gram。)

$$
\begin{aligned}
\text{几何平均} &= \exp\left(\tfrac{1}{4}(0 - 0.182 - 0.511 - 1.386)\right) = \exp(-0.520) \approx 0.595 \\
BP &= e^{\,1 - 8/7} = e^{-0.143} \approx 0.867 \\
\mathrm{BLEU} &= 0.867 \times 0.595 \approx 0.515 \;\Rightarrow\; \mathbf{51.5}
\end{aligned}
$$

(已用 `evaluate.corpus_bleu` 验证: 输出 51.54。)

可以看到: 只漏了一个词 `sitting`, unigram 精确率仍是满分, 但它打断了跨过该位置的所有 2/3/4-gram, 高阶 $p_n$ 明显下降; 译文偏短又被 BP 扣了 13%。

### 截断为什么必要

```
参考: the cat is on the mat
候选: the the the the the the
```

不截断: unigram 精确率 = 6/6 = 1, 荒谬。截断后 `the` 最多算 2 次 (参考中出现 2 次): $p_1 = 2/6 \approx 0.33$。

## 4. 为什么要用 BLEU

1. **交叉熵 / 困惑度衡量不了真实翻译质量**: 它们在 teacher forcing 下逐词算, 每一步都喂了正确的前文; 而真正翻译时模型要自己生成整句 (误差会累积), 还要经过 greedy / beam search。BLEU 直接评价**解码出来的最终译文**。
2. **便宜、自动、可复现**: 人工评价又贵又慢; BLEU 只需数 n-gram, 几秒跑完整个测试集, 同样的输入永远得到同样的分数。
3. **与人工评价大体相关** (在语料级别), 能反映词的选择 (低阶 n-gram) 和语序流畅度 (高阶 n-gram)。
4. **行业标准**: 几乎所有机器翻译论文 (包括本项目复现的 Bahdanau et al. 2015) 都报告 BLEU, 便于横向比较。

## 5. 必须注意的细节

1. **语料级, 不是句子级平均**: 正确做法是把所有句子的命中数、n-gram 总数、长度**先累加**, 最后算一次 (本项目的实现就是这样)。先算每句 BLEU 再取平均会得到不同 (且偏低、噪声大) 的数值。
2. **任何一个 $p_n = 0$ 则 BLEU = 0**: 因为 $\log 0 = -\infty$。语料级很少发生, 但单句上很常见 (比如短句根本没有 4-gram 命中)。算句子级 BLEU 时要用平滑 (smoothing, 如 NLTK 的 `SmoothingFunction`)。本项目的实现在这种情况下直接返回 0。
3. **分词决定分数**: 同一批译文, 大小写、标点是否切开、是否 BPE 都会让 BLEU 相差几个点。**不同分词的 BLEU 不能直接比较**。本项目用自己的分词, 数值不能和论文直接比; 要与论文可比, 应在 detokenize 后用 `sacrebleu` 计算。
4. **BP 只惩罚过短, 不惩罚过长**: 过长的译文会通过精确率下降被间接惩罚。
5. **只认字面匹配**: 同义词 (`big` vs `large`)、合理改写都算错; 分数高也不保证语义正确 (改一个否定词 `not` 意思全反, BLEU 只掉一点)。所以 BLEU 适合比较**同一任务上的不同系统**, 不适合当作绝对质量的标尺。
6. **参考数量影响绝对值**: 多个参考译文时匹配更容易, 分数更高; 本项目每句只有一个参考。
7. **分数范围参考**: 以 100 分制, 随机输出接近 0; 通常 20~30 已是能看懂的译文, 40+ 属于高质量 (取决于语言对和数据集, 只能作粗略直觉)。

## 6. 代码

本项目的实现 (`evaluate.py`), 语料级 BLEU-4, 每句一个参考:

```python
def corpus_bleu(hyps, refs, max_n=4):
    match, total = [0] * max_n, [0] * max_n
    c = r = 0
    for h, ref in zip(hyps, refs):
        c, r = c + len(h), r + len(ref)                   # 长度先累加
        for n in range(1, max_n + 1):
            hc, rc = _ngrams(h, n), _ngrams(ref, n)
            match[n - 1] += sum(min(v, rc[g]) for g, v in hc.items())   # 截断计数
            total[n - 1] += max(len(h) - n + 1, 0)
    if c == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return 100 * bp * math.exp(log_p)
```

标准工具 (需要与论文可比时):

```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(hyp_strings, [ref_strings])   # 输入是 detokenize 后的字符串
print(bleu.score)
```
