import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


class RunsTestAnalyzer:
    """
    RunsTestAnalyzer 游程检验分析器

    本类用于评估时间序列的随机性，基于 **游程检验（Runs Test）** 方法。

    -----------------------------------
    🔍 什么是“游程检验”（Runs Test）？
    -----------------------------------
    - 游程：符号连续相同的一段子序列（如 + + + 或 - -）；
    - 随机序列正负号应频繁切换（游程数接近理论值）；
    - 趋势或结构性行为 → 游程数偏少或偏多；

    -----------------------------------
    ✅ 本类功能：
    -----------------------------------
    - 自动执行游程检验（符号标准化、游程统计）；
    - 返回 Z 值、P 值、游程数、理论期望；
    - 存储每段游程（起止索引、符号）；
    - 可视化符号序列 + 高亮游程变化点。

    示例：
    >>> analyzer = RunsTestAnalyzer(series)
    >>> z, p, runs, expected = analyzer.compute()
    >>> analyzer.plot()
    """

    def __init__(self, series, method="median"):
        self.series = pd.Series(series).dropna()
        self.method = method
        self.signs = None
        self.runs = None
        self.z_stat = None
        self.p_value = None
        self.expected_runs = None
        self.run_segments = []  # ⬅️ 每段游程（起点, 终点, 符号）

    def compute(self):
        """
        执行游程检验并返回统计量

        返回
        ----------
        tuple:
            (z_stat, p_value, runs, expected_runs)
        """
        x = self.series.values
        center = np.median(x) if self.method == "median" else np.mean(x)
        self.signs = np.where(x >= center, 1, 0)

        # 统计游程
        self.runs = 1 + np.sum(self.signs[1:] != self.signs[:-1])
        n1 = np.sum(self.signs == 1)
        n2 = np.sum(self.signs == 0)

        self.expected_runs = 1 + 2 * n1 * n2 / (n1 + n2)
        std_runs = np.sqrt(
            2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
        )
        self.z_stat = (self.runs - self.expected_runs) / std_runs
        self.p_value = 2 * (1 - norm.cdf(np.abs(self.z_stat)))

        # 记录每段游程结构
        self.run_segments = []
        curr_sign = self.signs[0]
        start = 0
        for i in range(1, len(self.signs)):
            if self.signs[i] != curr_sign:
                self.run_segments.append((start, i - 1, curr_sign))
                start = i
                curr_sign = self.signs[i]
        self.run_segments.append((start, len(self.signs) - 1, curr_sign))

        return self.z_stat, self.p_value, self.runs, self.expected_runs

    def plot(self):
        """
        可视化正负符号序列及游程结构
        """
        if self.signs is None:
            raise ValueError("请先调用 .compute()")

        plt.figure(figsize=(12, 3))
        colors = np.where(self.signs == 1, "tab:blue", "tab:orange")

        # 画符号点
        for i in range(len(self.signs)):
            plt.plot(i, 1, marker="o", color=colors[i])

        # 画游程段界线
        for start, end, sign in self.run_segments:
            plt.hlines(1, start, end, colors="gray", linewidth=2, alpha=0.4)
            plt.plot([start, end], [1, 1], marker="|", color="gray", markersize=10)

        plt.title(f"游程检验图示 | 游程数 = {self.runs}, Z = {self.z_stat:.2f}, P = {self.p_value:.4f}")
        plt.yticks([])
        plt.xlabel("时间点")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
