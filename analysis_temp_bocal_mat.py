#!/usr/bin/env python3
"""
=============================================================================
SCRIPT 2 — Advanced Exploratory Data Analysis of temp_bocal.mat
=============================================================================
Dataset : Sea-water temperature at El Bocal (IEO, Santander)
Source  : MATLAB .mat file — structured variable 'bocal' (2192 × 1)
Period  : Presumed daily measurements spanning ~6 years (2012–2017)

Author  : Data-Science & Sensor Engineering Pipeline
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import io as sio
from scipy import signal, stats
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────
MAT_PATH = "temp_bocal.mat"
OUTPUT_DIR = "outputs"


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
FIGURE_DPI = 150
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (14, 5),
})


# ===========================================================================
#  1. INITIAL INSPECTION PHASE
# ===========================================================================
def inspect_mat_file(filepath: str) -> dict:
    """
    Load the .mat file and print the complete internal structure:
    key names, types, dtypes, shapes — including MATLAB metadata headers.
    """
    try:
        mat_data = sio.loadmat(filepath)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {filepath}")
    except Exception as e:
        sys.exit(f"[ERROR] Could not read .mat file: {e}")

    print("=" * 72)
    print("  PHASE 1 — .mat File Internal Structure Inspection")
    print("=" * 72)
    print(f"  File: {filepath}\n")

    for key, val in mat_data.items():
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            print(f"  Key: {key:30s}  |  type: {type(val).__name__:15s}  "
                  f"|  dtype: {str(val.dtype):12s}  |  shape: {val.shape}")
        else:
            val_repr = repr(val)[:80]
            print(f"  Key: {key:30s}  |  type: {type(val).__name__:15s}  "
                  f"|  value: {val_repr}")
    print()

    return mat_data


# ===========================================================================
#  2. DATA LOADING & PREPARATION
# ===========================================================================
def load_data(mat_data: dict) -> pd.DataFrame:
    """
    Extract the 'bocal' array from the loaded .mat dictionary.
    Since the .mat file has no explicit date vector, we reconstruct
    a daily date index starting on 01/01/2012 (consistent with the
    companion temp_bocal.txt file).

    Physical context: the 'bocal' variable stores daily SST readings
    from the IEO El Bocal coastal station.
    """
    if "bocal" not in mat_data:
        sys.exit("[ERROR] Expected variable 'bocal' not found in .mat file.")

    raw = mat_data["bocal"].flatten()
    N = len(raw)
    print(f"[INFO] 'bocal' array: {N} elements, dtype={raw.dtype}")

    # Build a daily date index starting 01/01/2012
    start_date = pd.Timestamp("2012-01-01")
    dates = pd.date_range(start=start_date, periods=N, freq="D")

    df = pd.DataFrame({"Temperatura": raw}, index=dates)
    df.index.name = "Fecha"

    # Replace zeros or extreme values with NaN if needed
    # (MATLAB sometimes stores 0 for missing data)
    # We flag any value outside a physically plausible range [5, 30] °C
    physically_implausible = (df["Temperatura"] < 5) | (df["Temperatura"] > 30)
    n_implausible = physically_implausible.sum()
    if n_implausible > 0:
        print(f"[WARN] {n_implausible} values outside plausible SST range "
              f"[5, 30] °C — set to NaN")
        df.loc[physically_implausible, "Temperatura"] = np.nan

    # Check for NaN
    n_missing = df["Temperatura"].isna().sum()
    print(f"[INFO] Missing/NaN values: {n_missing} out of {N}")
    print(f"[INFO] Date range: {df.index.min().date()} → {df.index.max().date()}")

    return df


# ===========================================================================
#  3. IN-DEPTH ANALYSIS
# ===========================================================================
def compute_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns that enrich the analysis:
      - dT/dt        : first-difference rate of change (°C/day)
      - T_30d_mean   : 30-day rolling average (thermal inertia proxy)
      - T_anomaly    : deviation from the 30-day rolling mean
      - Month / Year : for aggregation
    """
    df = df.copy()
    df["dTdt"] = df["Temperatura"].diff()
    df["T_30d_mean"] = df["Temperatura"].rolling(30, center=True, min_periods=10).mean()
    df["T_anomaly"] = df["Temperatura"] - df["T_30d_mean"]
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    df["DayOfYear"] = df.index.dayofyear
    return df


def identify_critical_points(df: pd.DataFrame) -> dict:
    """
    Detect local maxima and minima (thermal peaks and troughs)
    using scipy.signal.argrelextrema with a prominence-based filter.
    Physical meaning: annual maxima correspond to late-summer SST peaks,
    minima to late-winter SST lows.
    """
    T = df["Temperatura"].interpolate().values
    # Use a wider order (15 days) to capture seasonal-scale extrema
    max_idx = signal.argrelextrema(T, np.greater, order=15)[0]
    min_idx = signal.argrelextrema(T, np.less, order=15)[0]

    result = {
        "local_maxima_dates": df.index[max_idx],
        "local_maxima_values": T[max_idx],
        "local_minima_dates": df.index[min_idx],
        "local_minima_values": T[min_idx],
    }

    # Heating / cooling rates around each extremum (±5 days gradient)
    heating_rates, cooling_rates = [], []
    for idx in max_idx:
        if idx >= 5 and idx < len(T) - 5:
            heating_rates.append((T[idx] - T[idx - 5]) / 5.0)
            cooling_rates.append((T[idx + 5] - T[idx]) / 5.0)
    result["mean_heating_rate"] = np.mean(heating_rates) if heating_rates else np.nan
    result["mean_cooling_rate"] = np.mean(cooling_rates) if cooling_rates else np.nan

    return result


def compute_monthly_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape data into a Year × Month pivot table and compute
    the inter-month correlation matrix.
    Physical rationale: high correlations between adjacent months
    reflect thermal inertia of the water mass; low correlations
    between distant months (e.g., Feb ↔ Aug) confirm the seasonal cycle.
    """
    pivot = df.pivot_table(
        values="Temperatura",
        index="Year",
        columns="Month",
        aggfunc="mean"
    )
    pivot.columns = [f"Month {m:02d}" for m in pivot.columns]
    corr = pivot.corr()
    return corr


def compute_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-by-year summary statistics for inter-annual comparison.
    """
    yearly = df.groupby("Year")["Temperatura"].agg(
        ["mean", "median", "std", "min", "max", "count"]
    )
    yearly["range"] = yearly["max"] - yearly["min"]
    return yearly


# ===========================================================================
#  4. VISUALISATIONS
# ===========================================================================
def plot_heatmap_daily(df: pd.DataFrame):
    """
    Heatmap: Year × Day-of-Year matrix of temperature.
    This reveals the annual SST cycle and inter-annual variability.
    Spatial analogue: each row is a "year-layer" in time-space.
    """
    pivot = df.pivot_table(
        values="Temperatura",
        index="Year",
        columns="DayOfYear",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.pcolormesh(
        pivot.columns, pivot.index, pivot.values,
        cmap="RdYlBu_r", shading="auto"
    )
    cbar = fig.colorbar(im, ax=ax, label="Temperature (°C)", pad=0.02)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Year")
    ax.set_title("SST Heatmap — Year × Day-of-Year (El Bocal, .mat data)")
    ax.set_yticks(pivot.index)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_heatmap_daily.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_correlation_heatmap(corr: pd.DataFrame):
    """
    Inter-month correlation heatmap — reveals seasonal coupling.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        square=True, ax=ax, vmin=-1, vmax=1,
        cbar_kws={"label": "Pearson r"}
    )
    ax.set_title("Inter-Month Temperature Correlation Matrix")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_correlation_heatmap.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_scatter_matrix_seasons(df: pd.DataFrame):
    """
    Scatter-plot matrix of seasonal mean temperatures (DJF, MAM, JJA, SON)
    to inspect inter-seasonal relationships across years.
    """
    season_map = {12: "DJF", 1: "DJF", 2: "DJF",
                  3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA",
                  9: "SON", 10: "SON", 11: "SON"}
    df_s = df.copy()
    df_s["Season"] = df_s["Month"].map(season_map)

    # Assign a "seasonal year" (DJF Dec belongs to next year's winter)
    df_s["SeasonYear"] = df_s["Year"]
    df_s.loc[df_s["Month"] == 12, "SeasonYear"] += 1

    seasonal_pivot = df_s.pivot_table(
        values="Temperatura",
        index="SeasonYear",
        columns="Season",
        aggfunc="mean"
    )[["DJF", "MAM", "JJA", "SON"]]  # ordered

    g = sns.pairplot(
        seasonal_pivot.dropna().reset_index(),
        vars=["DJF", "MAM", "JJA", "SON"],
        diag_kind="kde",
        plot_kws={"alpha": 0.7, "s": 60, "edgecolor": "white"},
        diag_kws={"fill": True, "alpha": 0.5},
        corner=True,
    )
    g.figure.suptitle("Seasonal Mean Temperature — Pair-Plot (Year-by-Year)", y=1.02)
    g.figure.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_scatter_matrix_seasons.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_critical_points(df: pd.DataFrame, crit: dict):
    """
    Time series with detected local maxima and minima overlaid.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df["Temperatura"], lw=0.6, color="steelblue", alpha=0.7,
            label="Daily T")
    ax.plot(df.index, df["T_30d_mean"], lw=1.3, color="navy", label="30-day mean")

    ax.scatter(crit["local_maxima_dates"], crit["local_maxima_values"],
               marker="^", color="red", s=50, zorder=5, label="Local maxima")
    ax.scatter(crit["local_minima_dates"], crit["local_minima_values"],
               marker="v", color="blue", s=50, zorder=5, label="Local minima")

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("SST with Critical Points — Maxima & Minima (.mat data)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_critical_points.png"), dpi=FIGURE_DPI)
    plt.show()


def plot_yearly_boxplot(df: pd.DataFrame):
    """
    Box-plot of temperature by year — for inter-annual comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    years = sorted(df["Year"].unique())
    data_by_year = [df.loc[df["Year"] == y, "Temperatura"].dropna().values for y in years]

    bp = ax.boxplot(data_by_year, labels=years, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5),
                    medianprops=dict(color="darkred", lw=2))
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Inter-Annual SST Variability — Box Plot (.mat data)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig9_yearly_boxplot.png"), dpi=FIGURE_DPI)
    plt.show()


# ===========================================================================
#  5. MAIN EXECUTION
# ===========================================================================
def main():
    # ── Create output directory & set up logging ──────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = TeeLogger(os.path.join(OUTPUT_DIR, "script2_log.txt"))
    sys.stdout = logger

    print("=" * 72)
    print("  SCRIPT 2 — EDA of temp_bocal.mat")
    print("=" * 72)

    # ── Phase 1: Inspect .mat structure ───────────────────────────────────
    mat_data = inspect_mat_file(MAT_PATH)

    # ── Phase 2: Load data ────────────────────────────────────────────────
    df = load_data(mat_data)

    # ── Phase 3: Derived variables ────────────────────────────────────────
    df = compute_derived_variables(df)

    # ── Phase 3a: Year-by-year stats ──────────────────────────────────────
    yearly = compute_yearly_stats(df)
    print("\n── Yearly Summary Statistics ──")
    print(yearly.to_string(float_format="%.2f"))

    # ── Phase 3b: Critical points ─────────────────────────────────────────
    crit = identify_critical_points(df)
    print(f"\n[INFO] Detected {len(crit['local_maxima_dates'])} local maxima, "
          f"{len(crit['local_minima_dates'])} local minima")
    print(f"  Mean heating rate near maxima: {crit['mean_heating_rate']:.3f} °C/day")
    print(f"  Mean cooling rate near maxima: {crit['mean_cooling_rate']:.3f} °C/day")

    # Print top annual maxima
    print("\n── Annual Temperature Peaks ──")
    for d, v in zip(crit["local_maxima_dates"], crit["local_maxima_values"]):
        if v > 20.0:  # Only show significant peaks
            print(f"  {d.date()}  →  {v:.1f} °C")

    # ── Phase 3c: Inter-month correlation ─────────────────────────────────
    corr = compute_monthly_correlation_matrix(df)

    # ── Phase 4: Plots ────────────────────────────────────────────────────
    print("\n[INFO] Generating plots …")
    plot_heatmap_daily(df)
    plot_correlation_heatmap(corr)
    plot_scatter_matrix_seasons(df)
    plot_critical_points(df, crit)
    plot_yearly_boxplot(df)

    print(f"\n[DONE] All figures and logs saved to '{OUTPUT_DIR}/' directory.")
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
