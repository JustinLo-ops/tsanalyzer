from tsanalyzer.periodicity.acf import ACFAnalyzer
import numpy as np

if __name__ == "__main__":
    n = 300
    t = np.arange(n)
    series = np.sin(2 * np.pi * t / 25) + 0.3 * np.random.randn(n)
    analyzer = ACFAnalyzer(series, nlags=60)
    lag, strength, acf_vals = analyzer.compute()
    print(f"主周期滞后: {lag}")
    print(f"滞后处的自相关系数: {strength:.4f}")
    analyzer.plot()
