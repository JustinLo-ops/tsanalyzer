from tsanalyzer.periodicity import FFTAnalyzer
import numpy as np

if __name__ == "__main__":
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    y = 3 * np.sin(2 * np.pi * 50 * t) + 2 * np.sin(2 * np.pi * 120 * t)

    analyzer = FFTAnalyzer(y, sampling_rate=5)
    peak_freq, period, strength, freqs, amps = analyzer.compute()

    print(f"主频率: {peak_freq:.2f} Hz")
    print(f"主周期: {period:.4f} 秒")
    print(f"峰值振幅: {strength:.4f}")

    analyzer.plot(show_log=True)
