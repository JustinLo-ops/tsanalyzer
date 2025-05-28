import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


class ShannonEntropyAnalyzer:
    """
    ShannonEntropyAnalyzer 香农熵分析器

    本类用于评估时间序列的随机性与不确定性，基于 [信息熵（Shannon Entropy）] 理论。

    -----------------------------------
    什么是“香农熵”？（Shannon Entropy）
    -----------------------------------
    - 熵衡量信息的平均不确定度；
    - 离散分布中，所有符号出现概率越均匀，熵越高；
    - 序列越可预测（偏斜、重复），熵越低；
    - 最大熵 = log2(类别数)，代表完全随机；最小熵 = 0，代表完全确定。

    -----------------------------------
    本类功能：
    -----------------------------------
    - 支持离散化处理（固定分箱 bin 数或自定义阈值）；
    - 自动计算香农熵值；
    - 提供可选归一化熵（最大值归一）；
    - 可视化频率分布与信息熵高亮显示。

    示例：
    >>> analyzer = ShannonEntropyAnalyzer(series, bins=8)
    >>> entropy, norm_entropy, counts = analyzer.compute()
    >>> analyzer.plot()
    """

    def __init__(self, series, bins=10):
        """
        初始化香农熵分析器

        参数
        ----------
        series : array-like
            一维时间序列数据
        bins : int or list
            离散化分箱方式（整数表示等宽分箱数，list 表示自定义分割点）
        """
        self.series = pd.Series(series).dropna()
        self.bins = bins
        self.counts = None
        self.probs = None
        self.entropy = None
        self.norm_entropy = None

    def compute(self):
        """
        计算香农熵（Shannon entropy）和归一化熵

        返回
        ----------
        tuple:
            (entropy, norm_entropy, counts)
        """
        x = self.series.values

        # 离散化
        if isinstance(self.bins, int):
            digitized = np.digitize(x, bins=np.histogram_bin_edges(x, bins=self.bins))
        else:
            digitized = np.digitize(x, bins=self.bins)

        counter = Counter(digitized)
        total = sum(counter.values())

        self.counts = dict(sorted(counter.items()))
        self.probs = np.array(list(self.counts.values())) / total

        # 熵计算
        self.entropy = -np.sum(self.probs * np.log2(self.probs + 1e-12))  # 避免 log(0)
        self.norm_entropy = self.entropy / np.log2(len(self.probs)) if len(self.probs) > 1 else 0

        return self.entropy, self.norm_entropy, self.counts

    def plot(self):
        """
        绘制离散分布频率直方图及信息熵标注
        """
        if self.counts is None:
            raise ValueError("请先调用 .compute()")

        keys = list(self.counts.keys())
        values = list(self.counts.values())

        plt.figure(figsize=(8, 4))
        bars = plt.bar(keys, values, color='teal', alpha=0.6)

        plt.title(f"离散频率分布 | 信息熵 = {self.entropy:.3f}, 归一熵 = {self.norm_entropy:.3f}")
        plt.xlabel("离散区间编码")
        plt.ylabel("频数")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
