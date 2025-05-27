from tsanalyzer.periodicity import STLAnalyzer
import pandas as pd
import numpy as np

if __name__ == "__main__":
    t = pd.date_range(start="2000-01-01", periods=730, freq="D")
    seasonal = 10 * np.sin(2 * np.pi * np.arange(730) / 365)
    trend = 0.05 * np.arange(730)
    noise = np.random.normal(scale=2, size=730)
    y = seasonal + trend + noise

    series = pd.Series(y, index=t)

    analyzer = STLAnalyzer(series, period=365)
    trend, seasonal, resid = analyzer.compute()

    print("趋势项样本：", trend.head())
    print("季节项样本：", seasonal.head())
    print("残差项样本：", resid.head())

    analyzer.plot()
