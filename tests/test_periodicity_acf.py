from tsanalyzer.periodicity import ACFAnalyzer
import numpy as np

if __name__ == "__main__":
    n = 300
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 25) + 0.2 * np.random.randn(n)

    analyzer = ACFAnalyzer(y, nlags=100, alpha=0.05, qstat=True)
    lag, strength, acf_vals, confint, qstats, pvals, multi_peaks = analyzer.compute()
    print(f"主周期滞后: {lag}")
    print(f"显著周期滞后点（多周期）:")
    for l, s in multi_peaks:
        print(f"  - lag = {l}, strength = {s:.3f}")
    analyzer.plot()
