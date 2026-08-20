"""
1. plot_rate_choropleth  — geographic map, districts colored by rate
2. plot_rate_matrix      — district x time-period matrix heatmap
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rate_matrix(
    long_df: pd.DataFrame,
    value_col: str = "rate_per_capita",
    title: str = "Crime rate per 10,000 people — by district and month",
    figsize=(16, 4),
    cmap = "viridis"
):
    """District (rows) x time period (columns) heatmap"""
    long_df = long_df.copy()
    long_df = long_df[long_df["district"] != "Total"].reset_index(drop=True)
    print(long_df.head(3))
    pivot = long_df.pivot_table(index="district", columns="period", values=value_col)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.astype("float64")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": value_col.replace("_", " ")},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Period")
    ax.set_ylabel("District")
    # Thin out x-tick labels if there are many periods, so they stay readable
    n_cols = len(pivot.columns)
    step = max(1, n_cols // 24)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_xticklabels([str(pivot.columns[i]) for i in range(0, n_cols, step)], rotation=90, fontsize=8)
    plt.tight_layout()
    return fig


def plot_rate_choropleth(
    districts_gdf,
    rate_df: pd.DataFrame,
    district_col: str = "district",
    value_col: str = "rate_per_capita",
    title: str = "Crime rate per 10,000 people — by district",
    cmap: str = "YlOrRd",
    figsize=(16, 5),
):
    merged = districts_gdf.merge(rate_df, on=district_col, how="left")

    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column=value_col,
        cmap=cmap,
        linewidth=0.6,
        legend=True,
        ax=ax,
    )
    for _, row in merged.iterrows():
        if row.geometry is not None:
            centroid = row.geometry.representative_point()
            ax.annotate(
                row[district_col], xy=(centroid.x, centroid.y),
                ha="center", fontsize=7, color="black",
            )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_axis_off()
    plt.tight_layout()
    return fig