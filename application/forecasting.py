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


def parse_period(col: str):
    col = str(col).strip()

    if re.fullmatch(r"\d{4}", col):
        return pd.Period(col, freq="Y"), "Y"
    else:
        try:
            return pd.Period(col, freq="M"), "M"
        except (ValueError, TypeError):
            pass

    return None, None


# 1. Reshape: wide (district x year columns) -> long (district, year, count)
def wide_to_long(df: pd.DataFrame, id_col = "Police District") -> pd.DataFrame:
    period_cols, periods, freq = [], [], None
    for col in df.columns:
        period, col_freq = parse_period(col)
        if period is not None:
            period_cols.append(col)
            periods.append(period)
            freq = col_freq  # assume a single file is consistently Y or M

    if isinstance(id_col, str):
        id_vars = [id_col]
    else:
        id_vars = list(id_col)

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=period_cols,
        var_name="period_str",
        value_name="count",
    )

    period_lookup = dict(zip(period_cols, periods))
    long_df["period"] = long_df["period_str"].map(period_lookup)
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce")
    if isinstance(id_col, str):
        long_df = long_df.rename(columns={id_col: "district"}).drop(columns=["period_str"])
    else:
        long_df = long_df.rename(columns={id_vars[0]: "district"}).drop(columns=["period_str"])
    long_df["district"] = long_df["district"].str.strip()
    long_df.attrs["freq"] = freq
    long_df["year"] = long_df["period"].apply(lambda p: p.year)
 
    return long_df.sort_values(["district", "period"]).reset_index(drop=True)


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

def build_features(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df = df.copy()
 
    lags = [1, 2, 3, 6, 12] if freq == "M" else [1, 2, 3]
    windows = [3, 6, 12] if freq == "M" else [2, 3]
    seasonal_lag = 12 if freq == "M" else None
 
    for lag in lags:
        df[f"lag_{lag}"] = df["count"].shift(lag)
 
    df["diff_1"] = df["count"].diff(1)
    if seasonal_lag:
        df[f"diff_{seasonal_lag}"] = df["count"].diff(seasonal_lag)
 
    for window in windows:
        df[f"rolling_mean_{window}"] = df["count"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["count"].shift(1).rolling(window).std()
        df[f"rolling_min_{window}"] = df["count"].shift(1).rolling(window).min()
        df[f"rolling_max_{window}"] = df["count"].shift(1).rolling(window).max()
 
    max_window = windows[-1]
    df[f"lag1_vs_roll{max_window}"] = df["lag_1"] / (df[f"rolling_mean_{max_window}"] + 1e-6)
    if seasonal_lag:
        df[f"lag{seasonal_lag}_vs_roll{max_window}"] = (
            df[f"lag_{seasonal_lag}"] / (df[f"rolling_mean_{max_window}"] + 1e-6)
        )
 
    if isinstance(df["period"].dtype, pd.PeriodDtype):
        period_dt = df["period"]
    else:
        period_dt = pd.PeriodIndex(df["period"], freq=freq)
    df["year"] = period_dt.year if hasattr(period_dt, "year") else period_dt.dt.year
 
    if freq == "M":
        month = period_dt.month if hasattr(period_dt, "month") else period_dt.dt.month
        df["month"] = month
        # Cyclical encoding avoids an artificial Dec->Jan discontinuity
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
 
    df["time_idx"] = np.arange(len(df))
 
    if seasonal_lag:
        df["yoy_ratio"] = df["count"] / (df[f"lag_{seasonal_lag}"] + 1e-6)
 
    return df
 
 
def get_feature_cols(df: pd.DataFrame) -> list:
    # diff_1/diff_12 (and similar) are computed as *current* count minus a
    # past count, leaking data
    exclude = {"district", "count", "period", "date", "yoy_ratio"}
    exclude |= {c for c in df.columns if c.startswith("diff_")}
    return [c for c in df.columns if c not in exclude]
 
 
def make_lag_features(long_df: pd.DataFrame) -> pd.DataFrame:
    freq = long_df.attrs.get("freq", "Y")
    long_df = long_df.sort_values(["district", "period"]).copy()
 
    pieces = [
        build_features(group.reset_index(drop=True), freq=freq)
        for _, group in long_df.groupby("district", sort=False)
    ]
    result = pd.concat(pieces, ignore_index=True)
    result.attrs["freq"] = freq
    return result
 


# 4. Forecast next year per district
def forecast_next_year(
    long_df: pd.DataFrame,
    n_periods_ahead: int = 1,
    min_periods_required: int = 4,
) -> pd.DataFrame:
    
    freq = long_df.attrs.get("freq", "Y")
    results = []
    last_period = long_df["period"].max()

    for district, group in long_df.groupby("district"):
        group = group.dropna(subset=["count"]).sort_values("period")
        if len(group) < min_periods_required:
            continue  # not enough history to fit a trend responsibly
 
        # Use an integer time index (ordinal) as the regression input —
        # works uniformly whether periods are years or months.
        time_idx = np.arange(len(group)).reshape(-1, 1)
        y = group["count"].values
        model = LinearRegression().fit(time_idx, y)
 
        future_time_idx = np.arange(len(group), len(group) + n_periods_ahead).reshape(-1, 1)
        preds = model.predict(future_time_idx)
        future_periods = [last_period + i for i in range(1, n_periods_ahead + 1)]
 
        for period, pred in zip(future_periods, preds):
            results.append({
                "district": district,
                "period": period,
                "forecast_count": max(0, round(pred)),  # counts can't go negative
                "trend_slope_per_period": round(model.coef_[0], 1),
            })
 
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["period_str"] = result_df["period"].astype(str)
    return result_df


def forecast_national_total(long_df: pd.DataFrame, n_years_ahead: int = 1) -> pd.DataFrame:
    """Same approach applied to the summed national series."""
    national = long_df.groupby("period", as_index=False)["count"].sum()
    national["district"] = "New Zealand (Total)"
    national.attrs["freq"] = long_df.attrs.get("freq", "Y")
    return forecast_next_year(national, n_periods_ahead=n_years_ahead, min_periods_required=4)


# --------------------------------------------------------------------------
# 5. Convenience: run the whole pipeline
# --------------------------------------------------------------------------
def run_forecast_pipeline(
    raw_df: pd.DataFrame,
    id_col = "Police District",
    partial_year: int | None = None,
    months_elapsed: int | None = None,
    n_years_ahead: int = 1,
):
    long_df = wide_to_long(raw_df, id_col=id_col)
    freq = long_df.attrs.get("freq", "Y")

    # if partial_year is not None and months_elapsed is not None:
    #     long_df = annualize_partial_year(long_df, partial_year, months_elapsed, method="exclude")

    long_df = make_lag_features(long_df)

    # district_forecast = forecast_next_year(long_df, n_periods_ahead=n_years_ahead)
    # national_forecast = forecast_national_total(long_df, n_years_ahead=n_years_ahead)

    return {
        "long_df": long_df,
        # "district_forecast": district_forecast,
        # "national_forecast": national_forecast,
        "freq": freq,
    }