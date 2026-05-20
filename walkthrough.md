# EDA Scripts for El Bocal Sea-Water Temperature Data

## Summary

Created two independent Python scripts for advanced Exploratory Data Analysis of temperature data from El Bocal (IEO, Santander), spanning 2012–2017 (daily measurements).

---

## Script 1: [analysis_temp_bocal_txt.py](file:///home/daniroldan/uni/Sensores/adolfo/analysis_temp_bocal_txt.py)

**Analyzes `temp_bocal.txt`** — tab-separated file with columns `Fecha` (DD/MM/YYYY) and `Temperatura` (°C).

### Features Implemented
| Feature | Description |
|---|---|
| **Data loading** | European date parsing, NaN coercion for missing/blank entries (28 found) |
| **Outlier detection** | IQR method with k=1.5; fences computed at [5.30, 26.90] °C — 0 outliers in this clean dataset |
| **dT/dt** | First-order finite difference (°C/day) with thermal shock threshold at ±1.5 °C/day |
| **Statistics** | Mean, median, variance, P10/P90, skewness, kurtosis, Shapiro-Wilk normality test |
| **Thermal stability** | Rolling 30-day std < 0.3 °C to detect steady-state periods |
| **FFT analysis** | Hanning-windowed FFT → dominant period at **365.3 days** (annual cycle confirmed) |

### Visualizations (4 figures)
1. **Time series** — daily temp + 30-day rolling mean ± 1σ envelope + outlier markers
2. **Histogram + KDE** — temperature distribution with normal reference curve
3. **dT/dt plot** — rate of change with thermal shock detection
4. **FFT spectrum** — power vs. period with annotated annual/semi-annual peaks

---

## Script 2: [analysis_temp_bocal_mat.py](file:///home/daniroldan/uni/Sensores/adolfo/analysis_temp_bocal_mat.py)

**Analyzes `temp_bocal.mat`** — MATLAB file containing variable `bocal` (2192×1 float64 array).

### Features Implemented
| Feature | Description |
|---|---|
| **MAT inspection** | Prints all keys, types, dtypes, shapes (including `__header__`, `__version__`, `__function_workspace__`) |
| **Data loading** | Reconstructs daily date index from 01/01/2012; plausibility filter [5, 30] °C |
| **Derived variables** | dT/dt, 30-day rolling mean, temperature anomaly, month/year/DOY columns |
| **Critical points** | `scipy.signal.argrelextrema` with order=15 → 16 maxima, 27 minima detected |
| **Heating/cooling rates** | Mean gradient (±5 days) around each peak |
| **Inter-month correlation** | Year×Month pivot → Pearson correlation matrix |
| **Seasonal decomposition** | DJF/MAM/JJA/SON mean pivot for pair-plot |

### Visualizations (5 figures)
1. **Year×Day-of-Year heatmap** — spatial-temporal SST pattern (`pcolormesh`)
2. **Inter-month correlation heatmap** — lower-triangle annotated matrix
3. **Seasonal scatter/pair-plot** — DJF vs MAM vs JJA vs SON with KDE diagonals
4. **Critical points overlay** — time series with local maxima (▲) and minima (▼)
5. **Yearly box-plot** — inter-annual variability comparison

---

## Validation

Both scripts were validated against the actual data:

```
Script1: 2192 records, 28 NaN, Mean=15.99°C, Std=2.96°C
  IQR fences: [5.30, 26.90] → 0 outliers
  FFT dominant period: 365.3 days ✅

Script2: bocal(2192,), Mean=15.99°C, Range=[10.80, 23.10]°C
  16 local maxima, 27 local minima ✅
```

## How to Run

```bash
cd /home/daniroldan/uni/Sensores/adolfo
python3 analysis_temp_bocal_txt.py    # Script 1
python3 analysis_temp_bocal_mat.py    # Script 2
```

> [!NOTE]
> Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn` — all installed in the current environment.
