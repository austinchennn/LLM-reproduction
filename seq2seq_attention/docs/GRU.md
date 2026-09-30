# GRU (Gated Recurrent Unit, 门控循环单元)

论文: *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (Cho et al. 2014)

在本项目中的位置: Seq2Seq 的编码器 (双向 GRU) 和解码器 (单向 GRU) 都用它。

## 1. 定义

GRU 是一种**循环神经网络 (RNN) 单元**: 按时间步依次读入序列 $x_1, x_2, \dots, x_T$, 每一步用当前输入和上一步的隐藏状态计算新的隐藏状态。

与普通 RNN 的区别是它多了两个**门 (gate)**:

- **更新门 $z$ (update gate)**: 决定这一步保留多少旧状态、写入多少新内容。
- **重置门 $r$ (reset gate)**: 决定计算新内容时参考多少旧状态。

门的值在 0 到 1 之间, 本质是逐维的"开合比例"。有了门, 网络可以自己学会**什么时候记住, 什么时候忘掉**。

## 2. 要解决的问题

### 普通 RNN

$$
h_t = \tanh(W x_t + U h_{t-1} + b)
$$

每一步都把旧状态完全重写一遍, 带来两个问题:

1. **长距离信息被冲掉**: 第 1 个词的信息要经过 $T$ 次 $\tanh(U\,\cdot)$ 变换才能到第 $T$ 步, 中间每一步都会覆盖一部分。
2. **梯度消失 / 爆炸**: 反向传播时

   $$
   \frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} \mathrm{diag}\left(1 - h_t^2\right) U
   $$

   是 $T-1$ 个矩阵的连乘。$\tanh$ 的导数 $\le 1$, 如果 $U$ 的最大奇异值 $< 1$, 乘积会指数级趋于 0 (梯度消失, 学不到长距离依赖); 如果 $> 1$, 可能指数级增大 (梯度爆炸)。

### GRU 的思路

让新状态是旧状态与新内容的**加权平均**, 而不是完全重写:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

当 $z_t \approx 0$ 时, $h_t \approx h_{t-1}$, 信息原样传递, 梯度也几乎原样传回 (见第 4 节)。网络可以学会在不重要的词上关闭更新门, 把重要信息保留很多步。

## 3. 公式

### 符号说明

设输入维度为 $d$, 隐藏状态维度为 $n$, $t$ 为时间步。

| 符号 | 形状 | 含义 |
|---|---|---|
| $x_t$ | $\mathbb{R}^{d}$ | 第 $t$ 步的输入, 本项目中是词的 embedding (解码器还会拼上 attention 的上下文向量) |
| $h_{t-1}$ | $\mathbb{R}^{n}$ | 上一步的隐藏状态, 即到 $t-1$ 为止的"记忆"; $h_0$ 通常为全 0 (本项目解码器的 $h_0$ 由编码器得到) |
| $h_t$ | $\mathbb{R}^{n}$ | 这一步输出的新隐藏状态, 既是输出, 也传给下一步 |
| $z_t$ | $(0,1)^{n}$ | 更新门 |
| $r_t$ | $(0,1)^{n}$ | 重置门 |
| $\tilde{h}_t$ | $(-1,1)^{n}$ | 候选状态 (candidate), 这一步想写入的新内容 |
| $W_z, W_r, W_h$ | $\mathbb{R}^{n \times d}$ | 作用在输入 $x_t$ 上的权重 |
| $U_z, U_r, U_h$ | $\mathbb{R}^{n \times n}$ | 作用在旧状态 $h_{t-1}$ 上的权重 (循环权重) |
| $b_z, b_r, b_h$ | $\mathbb{R}^{n}$ | 偏置 |
| $\sigma(\cdot)$ | | sigmoid, $\sigma(a) = \frac{1}{1 + e^{-a}}$, 输出 $(0, 1)$, 用作门 |
| $\tanh(\cdot)$ | | 输出 $(-1, 1)$, 用作内容 |
| $\odot$ | | 逐元素相乘 (Hadamard 积) |

所有时间步**共享同一组参数** ($W, U, b$), 所以参数量与序列长度无关。

### 四个公式

$$
\begin{aligned}
z_t &= \sigma\left(W_z x_t + U_z h_{t-1} + b_z\right) && \text{(1) 更新门} \\
r_t &= \sigma\left(W_r x_t + U_r h_{t-1} + b_r\right) && \text{(2) 重置门} \\
\tilde{h}_t &= \tanh\left(W_h x_t + U_h \left(r_t \odot h_{t-1}\right) + b_h\right) && \text{(3) 候选状态} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t && \text{(4) 新状态}
\end{aligned}
$$

逐个解释:

- **(1) 更新门 $z_t$**: 根据当前输入和旧状态, 为 $h$ 的每一维算出一个 0~1 的比例。它在 (4) 中控制新旧的混合比例。
- **(2) 重置门 $r_t$**: 形式和 $z_t$ 一样, 但参数独立, 作用不同: 它只在 (3) 中使用。
- **(3) 候选状态 $\tilde{h}_t$**: 和普通 RNN 的公式几乎一样, 唯一区别是旧状态先乘上 $r_t$。
  - $r_t \approx 0$: 候选状态几乎只由当前输入 $x_t$ 决定, 相当于"忽略过去, 重新开始", 适合遇到新短语、新从句。
  - $r_t \approx 1$: 退化为普通 RNN 的计算方式。
- **(4) 新状态 $h_t$**: 每一维是旧值 $h_{t-1}$ 和候选值 $\tilde{h}_t$ 的凸组合 (两个系数加起来为 1)。
  - $z_t \approx 0$: $h_t \approx h_{t-1}$, 保留旧记忆, 跳过这个输入。
  - $z_t \approx 1$: $h_t \approx \tilde{h}_t$, 用新内容覆盖。
  - 每一维的 $z$ 各不相同, 所以可以部分维度保留、部分维度更新。

### 两个门的分工

| | 更新门 $z$ | 重置门 $r$ |
|---|---|---|
| 用在哪 | 公式 (4), 最终输出 | 公式 (3), 计算候选时 |
| 控制什么 | 旧状态**保留**多少 | 旧状态**参与计算新内容**多少 |
| 学到的倾向 | 负责长期依赖的维度, $z$ 经常接近 0 | 负责短期依赖的维度, $r$ 经常变化 |

Cho 2014 的原话: 学到短期依赖的单元, 重置门往往很活跃; 学到长期依赖的单元, 更新门往往很活跃 (大部分时间关着)。

### 小例子

句子: `the cat , which was very hungry , ate`

- 读到 `cat`: 某几维的 $z$ 打开, 写入"主语是单数"。
- 读从句 `which was very hungry`: 这几维的 $z \approx 0$, 信息不动; 其他维度正常更新, 处理从句内容。
- 读到 `ate`: 那几维的信息还在, 可以用来判断动词形式。

### 数值例子 ($n = 1$)

设 $h_{t-1} = 0.8$, 算出 $z_t = 0.1$, $r_t = 0.5$, $\tilde{h}_t = -0.6$:

$$
h_t = (1 - 0.1) \times 0.8 + 0.1 \times (-0.6) = 0.72 - 0.06 = 0.66
$$

更新门很小, 所以新状态基本保持在旧值附近。如果 $z_t = 0.9$, 则 $h_t = 0.08 - 0.54 = -0.46$, 几乎被新内容覆盖。

### 参数量

三组 $(W, U, b)$:

$$
3 \times \left(nd + n^2 + n\right)
$$

本项目编码器 $d = 256$, $n = 512$, 单方向 $3 \times (131072 + 262144 + 512) \approx 118$ 万, 双向 $\approx 236$ 万。

## 4. 为什么能缓解梯度消失

只看公式 (4) 中 $h_{t-1}$ 直接传到 $h_t$ 的那一项:

$$
\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}(1 - z_t) + \big(\text{通过 } z_t,\ r_t,\ \tilde{h}_t \text{ 的其他项}\big)
$$

第一项是一条**加法直通路径**: 当 $z_t \approx 0$ 时, 它接近单位矩阵 $I$, 连乘很多步也不会指数衰减。普通 RNN 没有这一项, 只能通过 $\mathrm{diag}(1-h_t^2)\,U$ 连乘。

这和 ResNet 的残差连接 $y = x + F(x)$、LSTM 的细胞状态 $c_t = f_t \odot c_{t-1} + \dots$ 是同一个思想: **用加法而不是重写来传递信息**, 让梯度有一条近似恒等的通路。后来 Transformer 里的残差连接也是如此。

注意: 门只缓解**梯度消失**。**梯度爆炸**仍然可能发生, 所以训练时还要做梯度裁剪 (本项目 `clip_grad_norm_(1.0)`)。

## 5. 与 LSTM 对比

### LSTM 的公式 (Hochreiter & Schmidhuber 1997, 加上遗忘门的版本)

$$
\begin{aligned}
i_t &= \sigma\left(W_i x_t + U_i h_{t-1} + b_i\right) && \text{输入门} \\
f_t &= \sigma\left(W_f x_t + U_f h_{t-1} + b_f\right) && \text{遗忘门} \\
o_t &= \sigma\left(W_o x_t + U_o h_{t-1} + b_o\right) && \text{输出门} \\
\tilde{c}_t &= \tanh\left(W_c x_t + U_c h_{t-1} + b_c\right) && \text{候选细胞状态} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{细胞状态} \\
h_t &= o_t \odot \tanh(c_t) && \text{隐藏状态 (输出)}
\end{aligned}
$$

### 对应关系

| LSTM | GRU | 说明 |
|---|---|---|
| 遗忘门 $f_t$ | $1 - z_t$ | 保留多少旧信息 |
| 输入门 $i_t$ | $z_t$ | 写入多少新信息 |
| ($f$ 与 $i$ 相互独立) | (两者绑定, 相加为 1) | GRU 规定"留得多就写得少", LSTM 可以同时留很多又写很多 |
| 输出门 $o_t$ | 无 | GRU 把整个状态直接输出 |
| 无 | 重置门 $r_t$ | GRU 在计算候选时控制是否参考过去 |
| 细胞状态 $c_t$ + 隐藏状态 $h_t$ | 只有 $h_t$ | GRU 把"记忆"和"输出"合并成一个 |

### 总体对比

| | LSTM | GRU |
|---|---|---|
| 提出时间 | 1997 | 2014 |
| 门数量 | 3 (输入、遗忘、输出) | 2 (更新、重置) |
| 状态 | 2 个 ($c$, $h$) | 1 个 ($h$) |
| 参数量 | $4(nd + n^2 + n)$ | $3(nd + n^2 + n)$, 约少 25% |
| 速度 | 较慢 | 较快 |
| 表达能力 | 稍强: 输入和遗忘独立; 细胞状态没有上界, 可以计数 | 稍弱, 但大多数任务上差别不大 |
| 适合 | 数据多、序列很长、需要精细记忆 | 数据较少、希望训练快 |

实证 (Chung et al. 2014, Jozefowicz et al. 2015): 两者在大多数任务上效果接近, 没有一方全面占优。LSTM 的遗忘门偏置初始化为 1 很重要, 加上这个技巧后 LSTM 通常略好或持平。

本项目用 GRU, 原因是 Bahdanau 论文用的就是它, 而且数据小、状态只有一个, 在解码器里与 attention 拼接更简单。

## 6. PyTorch 实现注意事项

PyTorch 的 `nn.GRU` 公式与上面的原论文写法有**两处不同**:

$$
\begin{aligned}
r_t &= \sigma\left(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}\right) \\
z_t &= \sigma\left(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}\right) \\
n_t &= \tanh\left(W_{in} x_t + b_{in} + r_t \odot \left(W_{hn} h_{t-1} + b_{hn}\right)\right) \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}
\end{aligned}
$$

1. **$z$ 的含义反过来**: PyTorch 中 $z$ 乘的是旧状态, 所以 $z \approx 1$ 表示保留旧记忆。只是记法不同, 学出来的门等价于 $1 - z$。
2. **重置门的位置**: PyTorch 先算 $W_{hn} h_{t-1}$ 再乘 $r_t$ (cuDNN 的实现方式, 可以把矩阵乘法合并加速); 原论文是先乘 $r_t$ 再乘 $U_h$。两者效果接近, 但数值不完全相同。
3. 每个门有两个偏置 ($b_{i\cdot}$, $b_{h\cdot}$), 这是为了兼容 cuDNN, 数学上等价于一个。

本项目用法:

```python
# 编码器: 整个序列一次算完
enc = nn.GRU(input_size=256, hidden_size=512, batch_first=True, bidirectional=True)
out, h_n = enc(packed_embeddings)          # out: [B, T, 1024], h_n: [2, B, 512]

# 解码器: 每一步输入要拼上 attention 的上下文向量, 只能逐步调用
dec_cell = nn.GRUCell(input_size=256 + 1024, hidden_size=512)
s_i = dec_cell(torch.cat([emb_prev, c_i], dim=-1), s_prev)   # [B, 512]
```

如果想手写一遍验证理解, 可以按第 3 节的四个公式写一个 `MyGRUCell`, 把权重从 `nn.GRUCell` 拷过来, 按第 6 节的写法对齐后比较输出是否一致。
