import numpy as np
from tsanalyzer.randomness import ShannonEntropyAnalyzer

if __name__ == "__main__":
    np.random.seed(0)
    y = np.random.normal(0, 1, 500)
    analyzer = ShannonEntropyAnalyzer(y, bins=10)
    entropy, norm_entropy, counts = analyzer.compute()
    print(f"信息熵: {entropy:.4f}")
    print(f"归一化熵: {norm_entropy:.4f}")
    analyzer.plot()
