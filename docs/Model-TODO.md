# 经典模型复现路线

按"每一步只多理解一个新东西"排序。

| 顺序 | 模型 / 论文 | 新学到的核心概念 | 建议数据 | 本地可行性 |
|---|---|---|---|---|
| ✅ | **word2vec** (2013) | 词向量、负采样 | text8 | 已完成 |
| 1 | **RNN / LSTM 语言模型** | 按顺序建模、预测下一个词、困惑度 (perplexity) | Tiny Shakespeare（1MB） | 几分钟 |
| 2 | **Seq2Seq + Attention** (Bahdanau 2014) | 编码器-解码器、**attention 的起源** | 小型翻译数据，比如英法句对 | 十几分钟 |
| 3 | **Transformer** (Attention Is All You Need, 2017) | self-attention、多头、位置编码 | 同上，做翻译 | 半小时左右 |
| 4 | **GPT-2** (2019) | decoder-only、自回归预训练、生成 | Tiny Shakespeare / OpenWebText 子集 | 小模型可以跑 |
| 5 | **BERT** (2018) | encoder-only、MLM、预训练 + 微调 | 小语料 + 一个分类任务 | 小模型可以跑 |
| 6 | **LLaMA 结构** | RMSNorm、RoPE、SwiGLU、KV cache | 同 GPT | 小模型可以跑 |
| 7 | **LoRA 微调 / SFT** | 在预训练模型上高效微调、指令对齐 | 加载现成的小模型，比如 Qwen 0.5B | M4 Max 可以跑 |
