import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


class STLAnalyzer:
    """
    STL 分解分析器（Seasonal-Trend decomposition using Loess）
    用于分解时间序列为 趋势 + 季节项 + 残差。
    """

    def __init__(self, series, period=365):
        """
        初始化 STL 分析器

        参数
        ----------
        series : array-like (带索引的 pd.Series 推荐)
            时间序列数据
        period : int
            季节周期（例如日数据中为 365，月数据中为 12）
        """
        if isinstance(series, (list, np.ndarray)):
            raise ValueError("请传入带时间索引的 pandas.Series")
        self.series = series.dropna()
        self.period = period
        self.result = None
        self.trend = None
        self.seasonal = None
        self.resid = None

    def compute(self):
        """
        执行 STL 分解

        返回
        ----------
        tuple:
            (trend, seasonal, residual)
        """
        stl = STL(self.series, seasonal=self.period)
        self.result = stl.fit()

        self.trend = self.result.trend
        self.seasonal = self.result.seasonal
        self.resid = self.result.resid

        return self.trend, self.seasonal, self.resid

    def plot(self):
        """
        绘制 STL 分解图，包括：
        原始时间序列、趋势项、季节项、残差项
        """
        if self.result is None:
            raise ValueError("请先调用 .compute()")

        t = self.series.index

        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axs[0].plot(t, self.series, label="原始")
        axs[0].set_title("原始时间序列")
        axs[0].grid(True)

        axs[1].plot(t, self.trend, label="趋势", color="blue")
        axs[1].set_title("趋势项（Trend）")
        axs[1].grid(True)

        axs[2].plot(t, self.seasonal, label="季节性", color="green")
        axs[2].set_title("季节项（Seasonal）")
        axs[2].grid(True)

        axs[3].plot(t, self.resid, label="残差", color="red")
        axs[3].set_title("残差项（Residual）")
        axs[3].grid(True)

        axs[3].set_xlabel("时间")

        plt.tight_layout()
        plt.show()
