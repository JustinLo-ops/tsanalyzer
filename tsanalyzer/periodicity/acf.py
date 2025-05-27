import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


class ACFAnalyzer:
    """
    自相关分析器（AutoCorrelation Function, ACF）

    本类用于检测时间序列中的周期性结构，基于 **自相关函数（ACF）**。

    --------------------------
    📌 什么是自相关（Autocorrelation）？
    --------------------------

    自相关描述的是一个时间序列在不同滞后（lag）下自身与自己的相关程度。
    简单地说，自相关就是把一个序列自身复制一份，向右移动 k 个时间点（称为“滞后”），
    然后计算原始序列与移动后的序列之间的相关性。

    公式定义（以零均值简化）：
        ACF(k) = cov(x_t, x_{t-k}) / var(x)

    直观含义：
    - ACF(0) 一定是 1，因为序列和自身完美相关；
    - 如果 ACF(k) 很高，说明当前值与过去 k 步的值之间有显著线性关系；
    - 如果在某个滞后处反复出现高自相关值，说明可能存在“周期性”重复行为。

    --------------------------
    📌 ACF 和周期检测的关系？
    --------------------------

    - 如果一个序列存在稳定的周期结构，那么它与滞后等于“周期长度”的自身将高度相关；
    - 举例：如果序列每 12 天重复一次，ACF 在滞后 12、24、36 处可能出现峰值；
    - ACF 可以通过观察“峰值滞后”来估计周期（主周期 = 最大峰值对应的滞后）。

    --------------------------
    📌 与 FFT、STL 的区别？
    --------------------------

    | 方法  | 原理          | 是否平稳要求 | 是否频域 | 适合用途        |
    |--------|---------------|---------------|-----------|------------------|
    | ACF    | 滞后相关性    | 是            | 否        | 周期初判、模型残差分析 |
    | FFT    | 正弦频率分解  | 是            | 是        | 精准频谱提取     |
    | STL    | 拟合趋势+季节 | 否            | 否        | 可视化成分分离   |

    --------------------------
    📌 本类功能概述
    --------------------------
    - 自动计算自相关系数（使用 statsmodels 的 acf 函数）
    - 可提取主周期滞后（peak_lag）及其强度（peak_strength）
    - 支持返回置信区间（alpha 参数）
    - 支持 Ljung-Box 检验（qstat=True）
    - 支持绘图，包括置信带 + 主滞后指示线

    使用示例：
    >>> analyzer = ACFAnalyzer(series, nlags=40, alpha=0.05, qstat=True)
    >>> lag, strength, acf_vals, confint, qstats, pvals = analyzer.compute()
    >>> analyzer.plot()

    推荐配合随机性检测和频域方法综合判断是否存在周期。
    """

    def __init__(self, series, nlags=60, alpha=0.05, adjusted=False, fft=True, qstat=False):
        """
        初始化 ACF 分析器

        参数
        ----------
        series : array-like
            一维时间序列（可为 list, np.array, pd.Series）
        nlags : int
            最大滞后阶数
        alpha : float
            显著性水平（置信区间）
        adjusted : bool
            是否调整自协方差除数
        fft : bool
            是否使用 FFT 计算
        qstat : bool
            是否计算 Ljung-Box 检验
        """
        self.series = pd.Series(series).dropna()
        self.nlags = nlags
        self.alpha = alpha
        self.adjusted = adjusted
        self.fft = fft
        self.qstat = qstat

        self.acf_vals = None
        self.confint = None
        self.qstat_vals = None
        self.pvalues = None
        self.peak_lag = None
        self.peak_strength = None
        self.multi_peaks = []

    def compute(self):
        """
        执行自相关分析，提取主周期滞后、自相关结构、置信区间等

        返回
        ----------
        tuple:
            (peak_lag, strength, acf_vals, confint, qstat, pvalues, multi_peaks)
        """
        results = acf(
            self.series,
            nlags=self.nlags,
            alpha=self.alpha,
            adjusted=self.adjusted,
            fft=self.fft,
            qstat=self.qstat
        )

        if self.alpha is not None and self.qstat:
            self.acf_vals, self.confint, self.qstat_vals, self.pvalues = results
        elif self.alpha is not None:
            self.acf_vals, self.confint = results
        elif self.qstat:
            self.acf_vals, self.qstat_vals, self.pvalues = results
        else:
            self.acf_vals = results

        self.peak_lag = int(np.argmax(self.acf_vals[1:]) + 1)
        self.peak_strength = float(self.acf_vals[self.peak_lag])

        acf_nonzero = self.acf_vals[1:]
        lags = np.arange(1, len(acf_nonzero) + 1)

        peaks, _ = find_peaks(acf_nonzero)

        self.multi_peaks = []
        if self.confint is not None:
            upper = self.confint[:, 1]
            for p in peaks:
                lag = lags[p]
                val = acf_nonzero[p]
                if val > upper[lag]:
                    self.multi_peaks.append((lag, float(val)))

        return (
            self.peak_lag,
            self.peak_strength,
            self.acf_vals,
            self.confint,
            self.qstat_vals,
            self.pvalues,
            self.multi_peaks
        )

    def plot(self):
        """
        绘制自相关函数图（包含置信区间）
        """
        if self.acf_vals is None:
            raise ValueError("请先调用 .compute()")

        lags = np.arange(len(self.acf_vals))
        plt.figure(figsize=(10, 4))
        plt.stem(lags, self.acf_vals)
        if self.confint is not None:
            upper = self.confint[:, 1]
            lower = self.confint[:, 0]
            plt.fill_between(lags, lower, upper, color='blue', alpha=0.2, label="置信区间")
        for lag, strength in self.multi_peaks:
            plt.axvline(lag, color="orange", linestyle=":", linewidth=1.2, alpha=0.7)
        plt.axvline(self.peak_lag, color="red", linestyle="--", label=f"主周期滞后 = {self.peak_lag}")
        plt.title("自相关函数（ACF）")
        plt.xlabel("滞后阶数 (lag)")
        plt.ylabel("自相关系数")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
