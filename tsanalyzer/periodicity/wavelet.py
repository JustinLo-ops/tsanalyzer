import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Microsoft YaHei"


class WaveletAnalyzer:
    """
    WaveletAnalyzer 小波周期分析器

    本类使用 [离散小波变换（DWT）] 对时间序列进行多尺度分解，
    并通过能量谱识别周期性结构的主尺度。适用于周期估计、降噪、周期成分提取等。

    -----------------------------
    小波分析的核心原理：
    -----------------------------
    - 小波变换将信号分解为多个频带（多尺度），每一层代表一个时间-频率分量；
    - 每一层 detail 系数（cD）包含不同频率区间的振荡成分；
    - 层级越高，对应的频率越低、周期越长；
    - 通过比较各层的能量大小，可以识别出周期性成分最显著的尺度层级。

    -----------------------------
    支持功能：
    -----------------------------
    - 自定义小波类型和边界模式；
    - 提取每一层的能量及主尺度；
    - 支持周期长度估计（需指定采样频率）；
    - 支持指定层的小波成分重构；
    - 支持绘图可视化（能量谱 + 主尺度标注）；

    -----------------------------
    参数说明：
    -----------------------------
    series : array-like
        一维时间序列数据
    wavelet : str or pywt.Wavelet
        使用的小波名称（如 'db4'、'haar'、'sym5' 等）
    max_level : int or None
        最大分解层数。若为 None，将根据数据自动计算
    mode : str
        边界延拓模式（常用有 'symmetric'、'periodization'、'zero'）

    示例用法：
    ----------
    >>> import numpy as np
    >>> t = np.arange(0, 300)
    >>> y = np.sin(2 * np.pi * t / 50) + 0.5 * np.random.randn(300)

    >>> analyzer = WaveletAnalyzer(y, wavelet='db4', mode='symmetric')
    >>> level, energy, periods = analyzer.compute(sampling_rate=1)
    >>> analyzer.plot()
    >>> reconstructed = analyzer.reconstruct(level)
    """

    def __init__(self, series, wavelet='db4', max_level=None, mode='symmetric'):
        self.series = np.asarray(series)
        self.wavelet = wavelet
        self.mode = mode
        self.max_level = max_level
        self.coeffs = None
        self.level_energy = None
        self.period_estimates = None
        self.dominant_level = None

    def compute(self, sampling_rate=1.0):
        """
        执行离散小波分解，并提取主周期层与能量信息

        参数：
        ----------
        sampling_rate : float
            采样频率（单位 Hz）。用于换算每个尺度层对应的周期。

        返回：
        ----------
        tuple:
            - dominant_level : int，主周期对应的层级（从 1 开始）
            - level_energy : List[float]，每层小波能量（除去近似层）
            - period_estimates : List[float]，每层对应的估计周期（单位：时间）

        注：周期计算公式为  估计周期 ≈ 2^j / f_s ，其中 j 为层级，f_s 为采样频率
        """
        self.coeffs = pywt.wavedec(self.series, wavelet=self.wavelet,
                                   mode=self.mode, level=self.max_level)
        detail_coeffs = self.coeffs[1:]  # 去掉第 0 层的近似系数
        self.level_energy = [np.sum(np.square(c)) for c in detail_coeffs]
        self.dominant_level = int(np.argmax(self.level_energy) + 1)

        # 周期估计（j=1 表示最低层，频率最高）
        self.period_estimates = [2 ** (i + 1) / sampling_rate for i in range(len(detail_coeffs))]

        return self.dominant_level, self.level_energy, self.period_estimates

    def reconstruct(self, level):
        """
        重构指定小波尺度层级的信号

        参数：
        ----------
        level : int
            要保留的细节层级（从 1 开始）

        返回：
        ----------
        recon : ndarray
            只保留第 level 层细节的重构信号
        """
        if self.coeffs is None:
            raise ValueError("请先调用 .compute()")

        coeffs_copy = [np.zeros_like(c) for c in self.coeffs]
        coeffs_copy[0] = np.zeros_like(self.coeffs[0])  # 近似系数设为 0
        coeffs_copy[level] = self.coeffs[level]  # 只保留指定层
        recon = pywt.waverec(coeffs_copy, wavelet=self.wavelet, mode=self.mode)
        return recon[:len(self.series)]  # 裁剪与原始长度一致

    def plot(self):
        """
        绘制每层小波能量分布图，并标记主尺度
        """
        if self.level_energy is None:
            raise ValueError("请先调用 .compute()")

        levels = range(1, len(self.level_energy) + 1)
        plt.figure(figsize=(8, 4))
        plt.bar(levels, self.level_energy, color='skyblue')
        plt.axvline(self.dominant_level, color="red", linestyle="--", label=f"主尺度 = Level {self.dominant_level}")
        plt.title("小波分解能量分布")
        plt.xlabel("小波尺度 Level")
        plt.ylabel("能量")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
