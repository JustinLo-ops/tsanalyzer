import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


class LombScargleAnalyzer:
    """
    使用 Lomb-Scargle 周期图分析周期性，支持非等间隔时间序列。
    """

    def __init__(self, t, y, min_freq=0.01, max_freq=1.0, resolution=1000):
        """
        初始化分析器

        参数
        ----------
        t : array-like
            时间点（可以是不规则间隔）
        y : array-like
            对应时间点的数值
        min_freq : float
            最小频率
        max_freq : float
            最大频率
        resolution : int
            频率网格的密度
        """
        self.t = np.asarray(t)
        self.y = np.asarray(y) - np.mean(y)
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resolution = resolution
        self.freqs = None
        self.power = None
        self.peak_freq = None
        self.period = None

    def compute(self):
        """
        计算 Lomb-Scargle 周期图

        返回
        ----------
        tuple: (peak_freq, period, freqs, power)
        """
        self.freqs = np.linspace(self.min_freq, self.max_freq, self.resolution)
        angular_freqs = 2 * np.pi * self.freqs
        self.power = lombscargle(self.t, self.y, angular_freqs)

        peak_idx = np.argmax(self.power)
        self.peak_freq = self.freqs[peak_idx]
        self.period = 1 / self.peak_freq if self.peak_freq > 0 else None

        return self.peak_freq, self.period, self.freqs, self.power

    def plot(self):
        """
        绘制周期图
        """
        if self.freqs is None or self.power is None:
            raise ValueError("请先调用 .compute()")

        plt.figure(figsize=(10, 5))
        plt.plot(1 / self.freqs, self.power)
        plt.axvline(self.period, color='red', linestyle='--', label=f"主周期 = {self.period:.2f}")
        plt.title("Lomb-Scargle 周期图")
        plt.xlabel("周期（Time Units）")
        plt.ylabel("功率谱强度")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
