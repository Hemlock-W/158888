"""
1. plot_rate_choropleth  — geographic map, districts colored by rate
2. plot_rate_matrix      — division x time-period matrix heatmap
3. plot_anzsoc_bar       — bar chart (descending order)
4. plot_stacked_bar      — stacked bar by Occurence Type Category 
5. plot_anzsoc_area      — area map, anzsoc x time-period stacked line
5. plot_anzsoc_treemap   — treemap, classification of OCC with count
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_rate_matrix(
    long_df:pd.DataFrame,
    value_col:str = "rate_per_capita",
    title:str = "Crime rate per 10,000 people — by district and month",
    cmap:str = "Viridis",
    xtitle = "Period",
    ytitle = "District",
):
    """District (rows) x time period (columns) heatmap"""

    long_df = long_df.copy()
    long_df = long_df[long_df["district"] != "Total"].reset_index(drop=True)

    pivot = long_df.pivot_table(
        index="district",
        columns="period",
        values=value_col
    )

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.astype("float64")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index,
            colorscale=cmap,
            colorbar=dict(
                title=value_col.replace("_", " ")
            ),
            hovertemplate=(
                "District: %{y}<br>"
                "Period: %{x}<br>"
                f"{value_col}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        height=max(400, len(pivot.index) * 25),
        margin=dict(l=80, r=20, t=60, b=80),
    )

    return fig

def plot_rate_choropleth( 
    districts_gdf, 
    rate_df:pd.DataFrame, 
    district_col:str = "district", 
    value_col:str = "rate_per_capita", 
    title:str = "Crime rate per 10,000 people — by district", 
    cmap:str = "YlOrRd", 
    figsize = (16, 5),
): 
    merged = districts_gdf.merge(rate_df, on=district_col, how="left") 
    fig, ax = plt.subplots(figsize=figsize) 
    merged.plot( column=value_col, cmap=cmap, linewidth=0.6, legend=True, ax=ax, ) 

    for _, row in merged.iterrows(): 
        if row.geometry is not None: 
            centroid = row.geometry.representative_point() 
            ax.annotate( row[district_col], xy=(centroid.x, centroid.y), ha="center", fontsize=4, color="black", ) 

    ax.set_title(title, fontsize=13, pad=12) 
    ax.set_axis_off() 

    plt.tight_layout() 
    return fig

def plot_anzsoc_bar(
    long_df:pd.DataFrame,
    value_col:str = "count",
    anzsoc_col:str = "district",
    title:str = "Crime classification - victim",
    xtitle:str = "Count",
    ytitle:str = "Type of Crime",
):
    long_df = long_df.copy()
    long_df = long_df[long_df[anzsoc_col] != "Total"].reset_index(drop=True)

    anzsoc_grouped = (long_df.groupby(anzsoc_col)[value_col]
                      .sum()
                      .sort_values(ascending=True)
                      .reset_index()
    )

    fig = px.bar(
        anzsoc_grouped,
        x=value_col,
        y=anzsoc_col,
        orientation="h",
    )

    fig.update_layout( 
        title=title,
        xaxis_title=xtitle, 
        yaxis_title=ytitle, 
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d")
    )

    return fig

def plot_stacked_bar(
    long_df:pd.DataFrame,
    value_col:str = "count",
    x_col:str = "district",
    color_col:str = "Occurrence Type Category",
    title:str = "Crime classification",
    xtitle:str = "Type of Crime",
    ytitle:str = "Count",
):
    melted = (
        long_df
        .reset_index()
        .melt(id_vars=color_col, var_name=x_col, value_name=value_col)
    )
    melted = melted[melted[x_col] != "index"].reset_index(drop=True)
 
    fig = go.Figure()
 
    totals = melted.groupby(color_col)[value_col].sum().sort_values(ascending=True)
 
    for i, category in enumerate(totals.index):
        subset = melted[melted[color_col] == category]
        fig.add_trace(go.Bar(
            x=subset[x_col],
            y=subset[value_col],
            name=str(category),
        ))
 
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
    )
 
    return fig
 
def plot_anzsoc_area(
    long_df:pd.DataFrame,
    period_col:str = "period",
    value_col:str = "count",
    anzsoc_col:str = "district",
    title:str = "Crime classification over time",
    xtitle:str = "Type of Crime",
    ytitle:str = "Count",
):
    long_df = long_df.copy()
    long_df = long_df[long_df[anzsoc_col] != "Total"].reset_index(drop=True)
    long_df[period_col] = long_df[period_col].dt.to_timestamp()
    long_df[anzsoc_col] = long_df[anzsoc_col].str.strip()
 
    fig = go.Figure()
    colors = px.colors.qualitative.Prism
 
    for i, category in enumerate(long_df[anzsoc_col].unique()):
        subset = long_df[long_df[anzsoc_col] == category].sort_values(period_col)
        fig.add_trace(go.Scatter(
            x=subset[period_col],
            y=subset[value_col],
            name=str(category),
            mode="lines",
            stackgroup="one",
            line=dict(color=colors[i % len(colors)]),
        ))
 
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
 
    return fig

def plot_anzsoc_treemap(
    long_df:pd.DataFrame,
    value_col:str = "count",
    label_col:str = "Occurrence Division",
    parent_col:str = "district",
    title:str = "Crime classification",
    xtitle:str = "Type of Crime",
    ytitle:str = "Count",
):
    long_df = long_df.copy()
    long_df = long_df[long_df[parent_col] != "Grand Total"]
    grouped = (
        long_df.groupby([parent_col, label_col])[value_col]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig = px.treemap(
        grouped, 
        path=[parent_col, label_col],
        values=value_col,
        color=value_col,
        color_continuous_scale="spectral",
    )

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )

    return fig
