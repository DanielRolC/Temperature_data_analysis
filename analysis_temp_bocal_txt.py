#!/usr/bin/env python3
"""
=============================================================================
SCRIPT 1 — Advanced Exploratory Data Analysis of temp_bocal.txt
=============================================================================
Dataset : Daily sea-water temperature at El Bocal (IEO, Santander)
Period  : 01/01/2012 – 31/12/2017
Columns : Fecha (DD/MM/YYYY)  |  Temperatura (°C)

Author  : Data-Science & Sensor Engineering Pipeline
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal, stats

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "temp_bocal.txt"
OUTPUT_DIR = "outputs"
FIGURE_DPI = 150


class TeeLogger:
    """Duplicate stdout to both console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (14, 5),
})


# ===========================================================================
#  1. DATA LOADING & CLEANING
# ===========================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """
    Read the tab-separated text file.
    • Parse European-format dates (DD/MM/YYYY).
    • Coerce non-numeric temperature values to NaN.
    • Sort chronologically and set the date as index.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            encoding="latin-1",          # handles possible special chars
            names=["Fecha", "Temperatura"],
            header=0,                     # skip original header
            skipinitialspace=True,
        )
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {filepath}")
    except Exception as e:
        sys.exit(f"[ERROR] Could not read file: {e}")

    # Parse dates (DD/MM/YYYY)
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")

    # Coerce temperature to numeric — blank or malformed entries become NaN
    df["Temperatura"] = pd.to_numeric(df["Temperatura"], errors="coerce")

    # Drop rows where date parsing failed entirely
    df.dropna(subset=["Fecha"], inplace=True)
    df.sort_values("Fecha", inplace=True)
    df.set_index("Fecha", inplace=True)

    print(f"[INFO] Loaded {len(df)} records  |  "
          f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"[INFO] Missing temperature values: {df['Temperatura'].isna().sum()}")

    return df


def detect_outliers_iqr(series: pd.Series, k: float = 1.5):
    """
    IQR-based outlier detection.
    Returns a boolean mask (True = outlier) and the fence boundaries.
    Physical rationale: sea-surface temperature has a well-defined annual
    range (~10 – 24 °C in the Bay of Biscay). Readings far outside the
    interquartile range likely represent sensor artefacts.
    """
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


def compute_dTdt(df: pd.DataFrame) -> pd.Series:
    """
    Compute the first-order finite difference dT/dt (°C/day).
    This is the daily rate of change — useful for detecting thermal shocks
    (abrupt heating/cooling events, e.g. upwelling or warm-water intrusions).
    """
    return df["Temperatura"].diff()  # ΔT per 1-day step


# ===========================================================================
#  2. ADVANCED STATISTICAL ANALYSIS
# ===========================================================================
def extract_metrics(df: pd.DataFrame) -> dict:
    """
    Compute key descriptive and physical statistics on the temperature series.
    """
    T = df["Temperatura"].dropna()

    metrics = {
        "N valid":          len(T),
        "Mean (°C)":        T.mean(),
        "Median (°C)":      T.median(),
        "Std Dev (°C)":     T.std(),
        "Variance (°C²)":   T.var(),
        "P10 (°C)":         T.quantile(0.10),
        "P90 (°C)":         T.quantile(0.90),
        "Min (°C)":         T.min(),
        "Max (°C)":         T.max(),
        "Range (°C)":       T.max() - T.min(),
        "Skewness":         T.skew(),
        "Kurtosis":         T.kurtosis(),
    }

    # ── Thermal stability: estimate time to reach "steady-state"
    # We define steady-state as the first contiguous window of 30 days
    # where the rolling standard deviation drops below 0.3 °C.
    # This captures periods of minimal thermal change (e.g., deep winter
    # or late summer plateaux).
    rolling_std = T.rolling(window=30, min_periods=15).std()
    stable_mask = rolling_std < 0.3
    if stable_mask.any():
        first_stable = stable_mask.idxmax()
        metrics["First stable period"] = str(first_stable.date())
    else:
        metrics["First stable period"] = "Not detected"

    return metrics


def fft_analysis(df: pd.DataFrame):
    """
    Fast Fourier Transform on the (linearly interpolated) temperature series.
    Purpose: detect dominant periodicities — e.g., the ~365-day annual cycle,
    or possible semi-annual or synoptic-scale oscillations.

    Returns:
        freqs   – array of frequencies (cycles/day)
        power   – normalized power spectrum (amplitude²)
        periods – corresponding period in days
    """
    T = df["Temperatura"].interpolate(method="linear").dropna().values
    N = len(T)
    dt = 1.0  # sampling interval = 1 day

    # Remove mean (DC component) to focus on oscillatory components
    T_detrended = T - T.mean()

    # Apply a Hanning window to reduce spectral leakage
    window = np.hanning(N)
    T_windowed = T_detrended * window

    fft_vals = np.fft.rfft(T_windowed)
    power = (2.0 / N) * np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(N, d=dt)

    # Convert to periods (skip DC at index 0)
    with np.errstate(divide="ignore"):
        periods = 1.0 / freqs

    return freqs[1:], power[1:], periods[1:]


# ===========================================================================
#  3. VISUALISATIONS
# ===========================================================================
def plot_timeseries(df: pd.DataFrame, outlier_mask: pd.Series):
    """
    Main time-series plot with a ±1-σ rolling envelope (30-day window)
    and outlier markers.
    """
    T = df["Temperatura"]
    roll_mean = T.rolling(window=30, center=True, min_periods=10).mean()
    roll_std = T.rolling(window=30, center=True, min_periods=10).std()

    fig, ax = plt.subplots(figsize=(15, 5))

    # Confidence band (mean ± 1σ)
    ax.fill_between(
        df.index, roll_mean - roll_std, roll_mean + roll_std,
        color="steelblue", alpha=0.20, label="±1σ envelope (30-day)")

    # Raw daily values
    ax.plot(df.index, T, linewidth=0.5, color="steelblue", alpha=0.6, label="Daily T")

    # 30-day rolling mean
    ax.plot(df.index, roll_mean, color="navy", linewidth=1.3, label="30-day mean")

    # Outliers
    if outlier_mask.any():
        ax.scatter(
            df.index[outlier_mask], T[outlier_mask],
            color="red", s=30, zorder=5, label=f"Outliers ({outlier_mask.sum()})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Sea-Water Temperature at El Bocal — Daily Evolution (2012–2017)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_timeseries.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_histogram_kde(df: pd.DataFrame):
    """
    Temperature distribution histogram with KDE overlay.
    Useful for assessing bimodality (summer vs. winter populations)
    and overall normality.
    """
    T = df["Temperatura"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(T, bins=50, density=True, color="steelblue", alpha=0.6,
            edgecolor="white", label="Histogram")

    # KDE via scipy
    kde = stats.gaussian_kde(T, bw_method="scott")
    x_grid = np.linspace(T.min() - 1, T.max() + 1, 500)
    ax.plot(x_grid, kde(x_grid), color="darkred", linewidth=2, label="KDE")

    # Normal reference curve
    mu, sigma = T.mean(), T.std()
    normal_pdf = stats.norm.pdf(x_grid, mu, sigma)
    ax.plot(x_grid, normal_pdf, "--", color="grey", linewidth=1.2,
            label=f"Normal ref (μ={mu:.1f}, σ={sigma:.1f})")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Probability density")
    ax.set_title("Temperature Distribution — Histogram + KDE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_histogram_kde.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_rate_of_change(df: pd.DataFrame, dTdt: pd.Series):
    """
    dT/dt plot — highlights sudden thermal gradients.
    Values > |1.5 °C/day| are flagged as potential thermal shocks
    (e.g., storm mixing, upwelling fronts).
    """
    fig, ax = plt.subplots(figsize=(15, 4))

    ax.bar(df.index, dTdt, width=1, color="steelblue", alpha=0.6, label="dT/dt")

    # Threshold lines for "thermal shocks" (±1.5 °C/day)
    shock_threshold = 1.5
    ax.axhline(shock_threshold, color="red", ls="--", lw=1, label=f"+{shock_threshold} °C/day")
    ax.axhline(-shock_threshold, color="red", ls="--", lw=1, label=f"−{shock_threshold} °C/day")

    # Highlight shock events
    shocks = dTdt.abs() > shock_threshold
    if shocks.any():
        ax.scatter(df.index[shocks], dTdt[shocks], color="red", s=25, zorder=5,
                   label=f"Thermal shocks ({shocks.sum()})")

    ax.set_xlabel("Date")
    ax.set_ylabel("dT/dt (°C/day)")
    ax.set_title("Temperature Rate of Change — Detection of Thermal Gradients")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_dTdt.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_fft(freqs, power, periods):
    """
    FFT power-spectrum plot — frequency domain analysis.
    Peaks indicate dominant periodic behaviours.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot in "period" space (more intuitive for geophysical data)
    valid = periods < 1500  # ignore very long periods beyond our window
    ax.semilogy(periods[valid], power[valid], color="steelblue", linewidth=0.8)

    # Mark known geophysical periods
    for p, label in [(365.25, "Annual (365 d)"), (182.6, "Semi-annual (183 d)")]:
        ax.axvline(p, color="red", ls="--", alpha=0.6, lw=1)
        ax.text(p + 10, power[valid].max() * 0.5, label,
                fontsize=9, color="red", rotation=0)

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("FFT Power Spectrum — Dominant Periodicities in SST")
    ax.set_xlim(1, 1500)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_fft_spectrum.png"), dpi=FIGURE_DPI)
    plt.show()


# ===========================================================================
#  4. MAIN EXECUTION
# ===========================================================================
def main():
    # ── Create output directory & set up logging ──────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = TeeLogger(os.path.join(OUTPUT_DIR, "script1_log.txt"))
    sys.stdout = logger

    print("=" * 72)
    print("  SCRIPT 1 — EDA of temp_bocal.txt")
    print("=" * 72)

    # ── Load & clean ──────────────────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Outlier detection ─────────────────────────────────────────────────
    outlier_mask, lower, upper = detect_outliers_iqr(df["Temperatura"].dropna())
    # Reindex mask to full df (handles NaN rows)
    outlier_mask = outlier_mask.reindex(df.index, fill_value=False)
    print(f"[INFO] IQR outlier fences: [{lower:.2f}, {upper:.2f}] °C  "
          f"→ {outlier_mask.sum()} outliers detected")

    # ── Rate of change ────────────────────────────────────────────────────
    dTdt = compute_dTdt(df)

    # ── Advanced metrics ──────────────────────────────────────────────────
    metrics = extract_metrics(df)
    print("\n── Statistical Summary ──")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} : {v:.4f}")
        else:
            print(f"  {k:25s} : {v}")

    # ── Normality test (Shapiro-Wilk on sub-sample — limit 5000) ────────
    T_clean = df["Temperatura"].dropna()
    if len(T_clean) > 5000:
        sample = T_clean.sample(5000, random_state=42)
    else:
        sample = T_clean
    stat, p_value = stats.shapiro(sample)
    print(f"\n  Shapiro-Wilk normality test: W={stat:.4f}, p={p_value:.2e}")
    print(f"  → {'Normal' if p_value > 0.05 else 'Non-normal'} distribution "
          f"(α=0.05)")

    # ── FFT analysis ──────────────────────────────────────────────────────
    freqs, power, periods = fft_analysis(df)
    # Identify the top-3 dominant periods
    top_idx = np.argsort(power)[::-1][:3]
    print("\n── Top-3 Dominant Periodicities (FFT) ──")
    for i, idx in enumerate(top_idx, 1):
        print(f"  #{i}  Period = {periods[idx]:.1f} days  "
              f"(freq = {freqs[idx]:.5f} cycles/day)  "
              f"Power = {power[idx]:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[INFO] Generating plots …")
    plot_timeseries(df, outlier_mask)
    plot_histogram_kde(df)
    plot_rate_of_change(df, dTdt)
    plot_fft(freqs, power, periods)

    print(f"\n[DONE] All figures and logs saved to '{OUTPUT_DIR}/' directory.")
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
