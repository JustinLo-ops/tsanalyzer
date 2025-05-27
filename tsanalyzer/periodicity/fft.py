import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"


class FFTAnalyzer:
    """
    快速傅里叶变换（FFT）分析器，适用于单变量时间序列的频谱分析。
    """

    def __init__(self, series, sampling_rate=1.0):
        """
        初始化 FFT 分析器

        参数
        ----------
        series : array-like
            时间序列数据（np.ndarray 或 pd.Series）
        sampling_rate : float
            采样频率（单位：Hz），默认为 1.0
        """
        self.series = np.asarray(series)
        self.sampling_rate = sampling_rate
        self.n = len(self.series)
        self.freqs = None
        self.amps = None
        self.peak_freq = None
        self.period = None
        self.strength = None

    def compute(self):
        """
        执行 FFT 计算并提取主频率、周期、振幅谱等信息。

        返回
        ----------
        tuple:
            (peak_freq, period, strength, frequencies, amplitudes)
        """
        x = self.series - np.mean(self.series)
        fft_values = fft(x)
        abs_y = np.abs(fft_values)
        norm_y = abs_y / self.n

        half_n = self.n // 2
        self.freqs = np.fft.fftfreq(self.n, d=1 / self.sampling_rate)[:half_n]
        self.amps = norm_y[:half_n]

        peak_index = np.argmax(self.amps[1:]) + 1
        self.peak_freq = self.freqs[peak_index]
        self.period = 1 / self.peak_freq if self.peak_freq != 0 else None
        self.strength = self.amps[peak_index]

        return (
            float(self.peak_freq),
            float(self.period),
            float(self.strength),
            self.freqs,
            self.amps,
        )

    def plot(self, show_log=False):
        """
        绘制原始波形图、FFT 单边振幅谱图和对数振幅谱（可选）

        参数
        ----------
        show_log : bool
            是否显示对数振幅谱图
        """
        if self.freqs is None or self.amps is None:
            raise ValueError("请先调用 .compute() 方法")

        fig, axs = plt.subplots(2 if not show_log else 3, 1, figsize=(10, 8))

        # 原始波形
        axs[0].plot(np.arange(self.n), self.series)
        axs[0].set_title("原始时间序列")
        axs[0].set_xlabel("时间 (样本点)")
        axs[0].set_ylabel("值")
        axs[0].grid(True)

        # 线性振幅谱
        axs[1].plot(self.freqs, self.amps)
        axs[1].axvline(self.peak_freq, color="red", linestyle="--", label=f"主频 = {self.peak_freq:.2f}Hz")
        axs[1].set_title("FFT 单边振幅谱")
        axs[1].set_xlabel("频率 (Hz)")
        axs[1].set_ylabel("振幅")
        axs[1].legend()
        axs[1].grid(True)

        # 对数谱（可选）
        if show_log:
            axs[2].plot(self.freqs, 20 * np.log10(self.amps + 1e-12))  # 避免 log(0)
            axs[2].set_title("FFT 对数振幅谱")
            axs[2].set_xlabel("频率 (Hz)")
            axs[2].set_ylabel("dB")
            axs[2].grid(True)

        plt.tight_layout()
        plt.show()
