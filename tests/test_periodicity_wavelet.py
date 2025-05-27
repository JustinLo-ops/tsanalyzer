import numpy as np
from tsanalyzer.periodicity import WaveletAnalyzer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = np.linspace(0, 10, 500)
    y = np.sin(2 * np.pi * t / 1.0) + 0.5 * np.sin(2 * np.pi * t / 0.25) + 0.2 * np.random.randn(len(t))

    analyzer = WaveletAnalyzer(y, wavelet='db4', mode='symmetric')

    level, energy, periods = analyzer.compute(sampling_rate=25)  # 假设 50Hz 采样率

    print(f"主尺度层: Level {level}")
    print("各层能量：", energy)
    print("周期估计：", [f"{p:.2f} 秒" for p in periods])

    # 可视化能量谱
    analyzer.plot()

    # 重构第主尺度层对应信号
    recon = analyzer.reconstruct(level)
    plt.figure(figsize=(10, 4))
    plt.plot(y, label="原始信号", alpha=0.5)
    plt.plot(recon, label=f"Level {level} 重构成分", linewidth=2)
    plt.title("小波主周期成分重构")
    plt.legend()
    plt.tight_layout()
    plt.show()