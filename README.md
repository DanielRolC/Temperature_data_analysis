# El Bocal Sea-Water Temperature Analysis

This directory contains two Python scripts designed for advanced Exploratory Data Analysis (EDA) of sea-water temperature data collected at El Bocal (IEO, Santander) between 2012 and 2017.

## Datasets

The data is provided in two formats:
1. `temp_bocal.txt`: A tab-separated text file containing daily temperature readings.
2. `temp_bocal.mat`: A MATLAB structured data file containing the temperature readings as a single array (`bocal`).

## Analysis Scripts

There are two independent Python scripts, each tailored to process one of the data formats and extract different advanced metrics and visualizations.

### 1. `analysis_temp_bocal_txt.py`
This script analyzes the `.txt` dataset.
- **Processing & Cleaning:** Handles European-format dates, filters out missing values, and detects outliers using the Interquartile Range (IQR) method.
- **Time Derivatives:** Calculates the daily rate of change ($dT/dt$) to identify sudden thermal gradients or thermal shocks (e.g., values > $\pm 1.5$ °C/day).
- **Statistical Analysis:** Computes mean, median, standard deviation, P10/P90 percentiles, skewness, kurtosis, and tests for normality (Shapiro-Wilk). Also estimates thermal stability periods.
- **Frequency Analysis:** Performs a Fast Fourier Transform (FFT) to identify dominant periodicities (e.g., annual cycles).
- **Visualizations:** Generates a time-series plot with a $\pm 1\sigma$ envelope, a histogram with a Kernel Density Estimate (KDE) overlay, a rate-of-change bar plot, and an FFT power spectrum plot.

### 2. `analysis_temp_bocal_mat.py`
This script analyzes the `.mat` dataset.
- **Internal Inspection:** Inspects and prints the internal structure, variables, and shapes stored within the `.mat` file.
- **Data Preparation:** Reconstructs the daily date index and filters physically implausible values.
- **Critical Points Detection:** Identifies local maxima (peaks) and minima (troughs) using prominence-based filtering, and computes heating/cooling rates around these extrema.
- **Correlation Analysis:** Computes inter-month Pearson correlations to evaluate seasonal coupling and thermal inertia.
- **Visualizations:** Produces a Year-by-Day-of-Year spatial-temporal heatmap, an inter-month correlation matrix heatmap, a seasonal scatter/pair-plot (DJF vs. MAM vs. JJA vs. SON), a time-series overlay of critical points, and a yearly box-plot for inter-annual comparison.

## Requirements

Ensure you have the required Python packages installed. You can install them using `pip`:

```bash
pip install numpy pandas matplotlib scipy seaborn
```

If you encounter a PEP 668 "externally managed environment" error on Debian/Ubuntu, you can install the system packages:
```bash
sudo apt install python3-numpy python3-pandas python3-matplotlib python3-scipy python3-seaborn
```
Or use a virtual environment.

## How to Run

Execute the scripts from the terminal:

```bash
python3 analysis_temp_bocal_txt.py
python3 analysis_temp_bocal_mat.py
```

## Outputs

Both scripts are configured to save all their outputs (generated figures and console logs) into a dedicated `outputs/` directory.

- **Figures:** Saved as high-resolution PNG files (`fig1_*.png` to `fig9_*.png`).
- **Logs:** Console outputs are captured in `outputs/script1_log.txt` and `outputs/script2_log.txt`.
