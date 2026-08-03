"""
Turns a wide-format crime export (one row per Police District, one column
per year) into a long time series, builds lag features, and forecasts the
next year's count per district (plus the national total).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------
# 1. Reshape: wide (district x year columns) -> long (district, year, count)
# --------------------------------------------------------------------------
def wide_to_long(df: pd.DataFrame, id_col: str="Police District") -> pd.DataFrame:
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]

    long_df = df.melt(
        id_vars=[id_col],
        value_vars=year_cols,
        var_name="year",
        value_name="count",
    )
    long_df["year"] = long_df["year"].astype(int)
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce")
    long_df = long_df.rename(columns={id_col: "district"})
    long_df["district"] = long_df["district"].str.strip()
    return long_df.sort_values(["district", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Handle a partial current year (data only runs to ~mid-year)
# --------------------------------------------------------------------------
def annualize_partial_year(
    long_df: pd.DataFrame,
    partial_year: int,
    months_elapsed: int,
    method: str = "exclude",
) -> pd.DataFrame:
    
    long_df = long_df.copy()
    if method == "exclude":
        return long_df[long_df["year"] != partial_year].reset_index(drop=True)
    elif method == "scale":
        mask = long_df["year"] == partial_year
        long_df.loc[mask, "count"] = long_df.loc[mask, "count"] * 12 / months_elapsed
        return long_df
    else:
        raise ValueError("method must be 'exclude' or 'scale'")


# --------------------------------------------------------------------------
# 3. Lag features (per district, since each district is its own series)
# --------------------------------------------------------------------------
def make_lag_features(long_df: pd.DataFrame, lags=(1, 2)) -> pd.DataFrame:
    long_df = long_df.sort_values(["district", "year"]).copy()
    grouped = long_df.groupby("district")["count"]

    for lag in lags:
        long_df[f"lag_{lag}"] = grouped.shift(lag)

    long_df["yoy_diff"] = grouped.diff(1)
    long_df["yoy_pct_change"] = grouped.pct_change(1)
    return long_df


# --------------------------------------------------------------------------
# 4. Forecast next year per district
# --------------------------------------------------------------------------
def forecast_next_year(
    long_df: pd.DataFrame,
    n_years_ahead: int = 1,
    min_years_required: int = 4,
) -> pd.DataFrame:
    results = []
    last_year = int(long_df["year"].max())

    for district, group in long_df.groupby("district"):
        group = group.dropna(subset=["count"]).sort_values("year")
        if len(group) < min_years_required:
            continue  # not enough history to fit a trend responsibly

        X = group[["year"]].values
        y = group["count"].values
        model = LinearRegression().fit(X, y)

        future_years = np.arange(last_year + 1, last_year + 1 + n_years_ahead)
        preds = model.predict(future_years.reshape(-1, 1))

        for fy, pred in zip(future_years, preds):
            results.append({
                "district": district,
                "year": int(fy),
                "forecast_count": max(0, round(pred)),  # counts can't go negative
                "trend_slope_per_year": round(model.coef_[0], 1),
            })

    return pd.DataFrame(results)


def forecast_national_total(long_df: pd.DataFrame, n_years_ahead: int = 1) -> pd.DataFrame:
    """Same approach applied to the summed national series."""
    national = long_df.groupby("year", as_index=False)["count"].sum()
    national["district"] = "New Zealand (Total)"
    return forecast_next_year(national, n_years_ahead=n_years_ahead, min_years_required=4)


# --------------------------------------------------------------------------
# 5. Convenience: run the whole pipeline
# --------------------------------------------------------------------------
def run_forecast_pipeline(
    raw_df: pd.DataFrame,
    id_col: str = "Police District",
    partial_year: int | None = None,
    months_elapsed: int | None = None,
    n_years_ahead: int = 1,
):
    long_df = wide_to_long(raw_df, id_col=id_col)

    if partial_year is not None and months_elapsed is not None:
        long_df = annualize_partial_year(long_df, partial_year, months_elapsed, method="exclude")

    long_df = make_lag_features(long_df)

    district_forecast = forecast_next_year(long_df, n_years_ahead=n_years_ahead)
    national_forecast = forecast_national_total(long_df, n_years_ahead=n_years_ahead)

    return {
        "long_df": long_df,
        "district_forecast": district_forecast,
        "national_forecast": national_forecast,
    }