"""超参配置: 改默认值就改这里; 命令行 --xxx 会覆盖同名字段"""

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple, get_args

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # 数据
    corpus: str = os.path.join(HERE, "data", "text8")  # 空格分词的纯文本; text8 不存在会自动下载
    max_tokens: Optional[int] = None  # 只用前 N 个词, 快速调试用
    min_count: int = 5                # 低于该词频的词丢弃
    subsample: float = 1e-4           # 高频词下采样阈值 t, 0 表示关闭

    # 模型
    mode: str = "sg"                  # "sg" (Skip-gram) / "cbow"
    dim: int = 100                    # embedding 维度, 一般 100~300
    window: int = 5                   # 最大窗口, 实际窗口 ~ U[1, window]
    neg: int = 5                      # 负样本数 K: 小语料 5~20, 大语料 2~5

    # 训练
    epochs: int = 3
    batch_size: int = 4096
    lr: Optional[float] = None        # 默认 sg=1e-3, cbow=2e-3 (SparseAdam)
    min_lr_ratio: float = 1e-4        # 学习率线性衰减到 lr * min_lr_ratio
    log_every: int = 1000
    seed: int = 42
    out_dir: Optional[str] = None     # 默认 output/<mode>_d<dim>

    # 评估
    probes: List[str] = field(default_factory=lambda: ["king", "france", "computer", "war", "one", "good"])
    analogies: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [("king", "man", "woman"), ("paris", "france", "germany")])

    def __post_init__(self):
        assert self.mode in ("sg", "cbow"), f"未知 mode: {self.mode}"
        if self.lr is None:
            self.lr = 1e-3 if self.mode == "sg" else 2e-3
        if self.out_dir is None:
            self.out_dir = os.path.join(HERE, "output", f"{self.mode}_d{self.dim}")

    @classmethod
    def from_args(cls, argv=None):
        """根据 dataclass 字段自动生成命令行参数 (列表类字段不暴露到命令行)"""
        p = argparse.ArgumentParser()
        for f in fields(cls):
            if f.name in ("probes", "analogies"):
                continue
            # Optional[int] -> int
            typ = next((t for t in get_args(f.type) if t is not type(None)), f.type)
            p.add_argument(f"--{f.name}", type=typ, default=None)
        args = {k: v for k, v in vars(p.parse_args(argv)).items() if v is not None}
        return cls(**args)
