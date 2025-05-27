# tsanalyzer

**tsanalyzer** is a Python toolkit for analyzing time series data. It aims to provide efficient and modular implementations for **periodicity detection**, **randomness estimation**, and **change point detection**, helping researchers and engineers extract structural insights from temporal data.

---

## ✨ Features

### ✅ Periodicity Detection (Implemented)
- `FFTAnalyzer`: Fast Fourier Transform for frequency spectrum analysis.
- `STLAnalyzer`: Seasonal-Trend decomposition using Loess (STL) for seasonal and trend separation.
- `ACFAnalyzer`: AutoCorrelation Function analysis for identifying periodic lags in time series.

### 🚧 In Progress
- Randomness metrics (entropy-based, chaos analysis, etc.)
- Change point detection (CUSUM, PELT, Bayesian Online Change Point Detection)
- Visualization utilities for exploratory data analysis

---

## 📦 Installation

You can install the latest version directly from GitHub:

```bash
pip install git+https://github.com/JustinLo-ops/tsanalyzer.git
```
Or clone the repository and install locally:
```bash
git clone https://github.com/JustinLo-ops/tsanalyzer.git
cd tsanalyzer
pip install .
```
---

## 🚀 Quick Start
```python
from tsanalyzer.periodicity import FFTAnalyzer, STLAnalyzer
import numpy as np
import pandas as pd

# FFT Example
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
y = 3 * np.sin(2 * np.pi * 50 * t) + 2 * np.sin(2 * np.pi * 120 * t)

fft_analyzer = FFTAnalyzer(y, sampling_rate=fs)
fft_analyzer.compute()
fft_analyzer.plot()

# STL Example
date_range = pd.date_range("2000-01-01", periods=730, freq="D")
y = 5 * np.sin(2 * np.pi * np.arange(730) / 365) + 0.05 * np.arange(730)
series = pd.Series(y, index=date_range)

stl_analyzer = STLAnalyzer(series, period=365)
stl_analyzer.compute()
stl_analyzer.plot()
```
