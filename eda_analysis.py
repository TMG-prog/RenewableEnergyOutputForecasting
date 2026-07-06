

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = r"C:\Users\ritar\ML Project\RenewableEnergyOutputForecasting\Project\data\cleaned\GlobalWeatherRepository_Cleaned_Optimized.csv"


OUTPUT_DIR = "reports/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded '{path}' — shape: {df.shape}\n")
    return df


# ----------------------------------------------------------------------
# 2. EXPLORE STRUCTURE
# ----------------------------------------------------------------------
def explore_structure(df):
    print("=" * 70)
    print("DATA STRUCTURE")
    print("=" * 70)
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    print("\nSummary statistics (numeric columns):")
    print(df.describe().T)

    # Save summary stats to a CSV for your report appendix
    df.describe().T.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
    print(f"\nSaved summary statistics -> {OUTPUT_DIR}/summary_statistics.csv\n")


# ----------------------------------------------------------------------
# Helper: detect useful column groups automatically
# ----------------------------------------------------------------------
def detect_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # try to find a datetime column
    datetime_col = None
    candidate_names = ["date", "last_updated", "timestamp", "datetime", "time"]
    for c in df.columns:
        if any(name in c.lower() for name in candidate_names):
            try:
                pd.to_datetime(df[c])
                datetime_col = c
                break
            except Exception:
                continue

    # try to find a categorical grouping column (for box plots)
    category_col = None
    candidate_cat_names = ["country", "location", "region", "season", "city"]
    for c in df.columns:
        if any(name in c.lower() for name in candidate_cat_names):
            category_col = c
            break

    print(f"Detected numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Detected datetime column: {datetime_col}")
    print(f"Detected categorical column for grouping: {category_col}\n")

    return numeric_cols, datetime_col, category_col


# ----------------------------------------------------------------------
# 3. CORRELATION ANALYSIS + HEATMAP
# ----------------------------------------------------------------------
def correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(min(1 + 0.6 * len(numeric_cols), 18),
                        min(1 + 0.6 * len(numeric_cols), 14)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap of Weather Variables", fontsize=14, pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved correlation heatmap -> {path}")

    # Extract and print the strongest relationships (excluding self-corr = 1.0)
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    print("\nTop 10 strongest correlations between variables:")
    print(corr_pairs.head(10))
    corr_pairs.head(20).to_csv(os.path.join(OUTPUT_DIR, "top_correlations.csv"))
    return corr


# ----------------------------------------------------------------------
# 4. HISTOGRAMS
# ----------------------------------------------------------------------
def plot_histograms(df, numeric_cols):
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution of {col}")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "histograms.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved histograms -> {path}")


# ----------------------------------------------------------------------
# 5. BOX PLOTS (outlier detection)
# ----------------------------------------------------------------------
def plot_boxplots(df, numeric_cols, category_col=None, max_categories=8):
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    use_category = (
        category_col is not None
        and df[category_col].nunique() <= max_categories
    )

    for i, col in enumerate(numeric_cols):
        if use_category:
            sns.boxplot(data=df, x=category_col, y=col, ax=axes[i])
            axes[i].tick_params(axis="x", rotation=45)
        else:
            sns.boxplot(y=df[col], ax=axes[i], color="lightseagreen")
        axes[i].set_title(f"Box Plot: {col}")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplots.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved box plots -> {path}")


# ----------------------------------------------------------------------
# 6. LINE CHARTS (time trend)
# ----------------------------------------------------------------------
def plot_line_charts(df, numeric_cols, datetime_col, max_vars=4):
    if datetime_col is None:
        print("No datetime column detected — skipping line charts. "
              "Add a date/time column name to CONFIG if one exists.")
        return

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col)

    # If there are many rows per timestamp (multiple locations), aggregate by day
    daily = df.set_index(datetime_col)[numeric_cols].resample("D").mean()

    vars_to_plot = numeric_cols[:max_vars]
    fig, axes = plt.subplots(len(vars_to_plot), 1,
                              figsize=(12, 3 * len(vars_to_plot)), sharex=True)
    if len(vars_to_plot) == 1:
        axes = [axes]

    for ax, col in zip(axes, vars_to_plot):
        ax.plot(daily.index, daily[col], color="darkorange")
        ax.set_title(f"{col} Over Time (daily average)")
        ax.set_ylabel(col)

    plt.xlabel("Date")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "line_charts.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved line charts -> {path}")


# ----------------------------------------------------------------------
# 7. SCATTER PLOTS 
# ----------------------------------------------------------------------
def plot_scatter_top_pairs(df, corr, top_n=4):
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    top_pairs = corr_pairs.head(top_n).index.tolist()

    if not top_pairs:
        print("Not enough numeric variable pairs for scatter plots.")
        return

    ncols = 2
    nrows = int(np.ceil(len(top_pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (x_col, y_col) in enumerate(top_pairs):
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.5, ax=axes[i])
        r_val = corr.loc[x_col, y_col]
        axes[i].set_title(f"{x_col} vs {y_col} (r = {r_val:.2f})")

    for j in range(len(top_pairs), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scatter_plots.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved scatter plots -> {path}")



def write_insights_summary(df, corr, numeric_cols):
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

    lines = []
    lines.append("EDA KEY INSIGHTS SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append(f"Numeric variables analyzed: {len(numeric_cols)}")
    lines.append("")
    lines.append("Strongest correlations found:")
    for (a, b), val in corr_pairs.head(5).items():
        direction = "positive" if val > 0 else "negative"
        lines.append(f"  - {a} & {b}: r = {val:.2f} ({direction})")
    lines.append("")
    lines.append("Variability (std dev) per variable — higher = more spread:")
    for col in numeric_cols:
        lines.append(f"  - {col}: std = {df[col].std():.2f}, "
                      f"mean = {df[col].mean():.2f}")
    lines.append("")
    lines.append("NOTE: Review boxplots.png for outliers to flag to the modeling team,")
    lines.append("and correlation_heatmap.png / scatter_plots.png for variables")
    lines.append("most likely to be predictive of renewable energy output potential.")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(os.path.join(OUTPUT_DIR, "insights_summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\nSaved insights summary -> {OUTPUT_DIR}/insights_summary.txt")



def main():
    df = load_data(DATA_PATH)
    explore_structure(df)
    numeric_cols, datetime_col, category_col = detect_columns(df)

    if len(numeric_cols) < 2:
        raise ValueError("Fewer than 2 numeric columns detected — check DATA_PATH "
                          "and that your CSV loaded correctly.")

    corr = correlation_heatmap(df, numeric_cols)
    plot_histograms(df, numeric_cols)
    plot_boxplots(df, numeric_cols, category_col)
    plot_line_charts(df, numeric_cols, datetime_col)
    plot_scatter_top_pairs(df, corr)
    write_insights_summary(df, corr, numeric_cols)

    print(f"\nAll done. Figures and summaries saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()