import numpy as np
from tsanalyzer.periodicity import LombScargleAnalyzer

if __name__ == "__main__":
    np.random.seed(0)
    t = np.sort(np.random.uniform(0, 100, 500))
    y = 3 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, size=len(t))

    analyzer = LombScargleAnalyzer(t, y, min_freq=0.01, max_freq=0.2)
    peak_freq, period, freqs, power = analyzer.compute()

    print(f"Detected peak frequency: {peak_freq:.4f} Hz")
    print(f"Corresponding period: {period:.2f} time units")

    analyzer.plot()
