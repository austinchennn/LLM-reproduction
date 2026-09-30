"""超参配置: 改默认值就改这里; 命令行 --xxx 会覆盖同名字段"""

import argparse
import os
from dataclasses import dataclass, fields
from typing import Optional, get_args

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # 数据
    data_path: str = os.path.join(HERE, "data", "eng-fra.txt")  # 每行 "英文 \t 法文"; 不存在会自动下载
    max_len: int = 15                 # 源句和目标句都不超过这么多词 (不含 <eos>)
    min_count: int = 2                # 训练集中出现少于该次数的词记为 <unk>
    max_pairs: Optional[int] = None   # 只用前 N 个句对, 快速调试用
    valid_ratio: float = 0.05
    test_ratio: float = 0.05

    # 模型
    emb_dim: int = 256                # E
    hid_dim: int = 512                # H, 编码器每个方向和解码器的 GRU 隐藏维度
    attn_dim: int = 512               # A, 加性注意力的中间维度
    maxout_dim: int = 256             # maxout 层输出维度 (pool 大小 2)
    dropout: float = 0.3
    no_attention: bool = False        # True: 关掉 attention, 解码器每步都看同一个固定上下文向量 (baseline)
    tie_weights: bool = False         # 目标 embedding 与输出层共享权重, 需要 maxout_dim == emb_dim

    # 训练
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    clip: float = 1.0                 # 梯度裁剪的 L2 范数上限
    teacher_forcing: float = 1.0      # 训练时用真实上一个词作为输入的概率
    lr_patience: int = 1              # valid loss 连续多少个 epoch 不降就把学习率减半
    log_every: int = 200
    seed: int = 42
    device: Optional[str] = None      # 默认自动: cuda > mps > cpu
    out_dir: str = os.path.join(HERE, "output")

    # 推理 / 评估
    beam_size: int = 5
    len_alpha: float = 0.7            # beam search 长度归一化: score = log p / len^alpha
    bleu_samples: int = 1000          # 每个 epoch 用多少条 valid 句子算 BLEU (greedy)

    def __post_init__(self):
        if self.tie_weights:
            assert self.maxout_dim == self.emb_dim, "tie_weights 需要 maxout_dim == emb_dim"
        if self.device is None:
            import torch
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def from_args(cls, argv=None, parser=None):
        """根据 dataclass 字段自动生成命令行参数; parser 可以预先加好脚本自己的参数"""
        p = parser or argparse.ArgumentParser()
        for f in fields(cls):
            typ = next((t for t in get_args(f.type) if t is not type(None)), f.type)
            if typ is bool:
                typ = lambda s: s.lower() in ("1", "true", "yes", "y")
            p.add_argument(f"--{f.name}", type=typ, default=None)
        ns = p.parse_args(argv)
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in vars(ns).items() if k in names and v is not None})
        return cfg, ns
