import numpy as np
from tsanalyzer.randomness import RunsTestAnalyzer

if __name__ == "__main__":
    np.random.seed(123)
    y = np.random.normal(0, 1, size=200)

    analyzer = RunsTestAnalyzer(y, method="median")
    z, p, runs, expected = analyzer.compute()
    print(f"Z统计量：{z:.3f}")
    print(f"P 值：{p:.4f}")
    print(f"实际游程数：{runs}，理论期望值：{expected:.2f}")
    analyzer.plot()

