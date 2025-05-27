import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


class ACFAnalyzer:
    """
    自相关分析器，用于通过 ACF 分析周期性。
    """

    def __init__(self, series, nlags=60):
        """
        初始化 ACF 分析器

        参数
        ----------
        series : array-like
            一维时间序列（可以是 list, np.array, pd.Series）
        nlags : int
            最大滞后阶数，用于计算自相关
        """
        self.series = pd.Series(series).dropna()
        self.nlags = nlags
        self.acf_vals = None
        self.peak_lag = None
        self.peak_strength = None

    def compute(self):
        """
        计算自相关函数并识别主周期滞后点

        返回
        ----------
        tuple:
            (peak_lag, strength, acf_array)
        """
        self.acf_vals = acf(self.series, nlags=self.nlags)
        self.peak_lag = int(np.argmax(self.acf_vals[1:]) + 1)
        self.peak_strength = float(self.acf_vals[self.peak_lag])

        return self.peak_lag, self.peak_strength, self.acf_vals

    def plot(self):
        """
        绘制自相关函数图
        """
        if self.acf_vals is None:
            raise ValueError("请先调用 .compute()")

        plt.figure(figsize=(10, 4))
        lags = np.arange(len(self.acf_vals))
        plt.stem(lags, self.acf_vals)
        plt.axvline(self.peak_lag, color="red", linestyle="--", label=f"主周期滞后 = {self.peak_lag}")
        plt.title("自相关函数（ACF）")
        plt.xlabel("滞后阶数 (lag)")
        plt.ylabel("自相关系数")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
