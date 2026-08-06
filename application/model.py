"""
1. scikit-learn regressor on lag features, forecast recursively
   (predict one step -> feed it back in as the next lag -> repeat)
2. Prophet, fit directly on the (date, count) series
3. statsmodels — Exponential Smoothing (ETS) or SARIMA

All three return {"model", "forecast", "metrics"}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from forecasting import make_lag_features

# Only pure lag_N columns are safe model inputs — they're strictly past
# values. diff_1 / pct_change_1 / yoy_diff / yoy_pct_change (also produced
# by make_lag_features) are computed as *current* count vs a past count,
# so they depend on the very value being forecast and would leak the
# target into the features. Don't use those for model training.
LAG_FEATURE_PREFIXES = ("lag_",)


def _lag_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in LAG_FEATURE_PREFIXES)]


def _to_timestamp_series(df: pd.DataFrame) -> pd.Series:
    return df["period"].apply(lambda p: p.to_timestamp())


# --------------------------------------------------------------------------
# 1. scikit-learn on lag features, with recursive multi-step forecasting
# --------------------------------------------------------------------------
def sklearn_forecast(
    long_df: pd.DataFrame,
    district: str,
    n_periods_ahead: int = 6,
    model=None,
    test_size: int = 6,
):
    """
    Trains a regressor on lag/diff features for one district, backtests it
    on the last `test_size` periods, then forecasts forward recursively:
    each predicted value becomes the lag input for the next step, since
    real future lags don't exist yet.
    """
    model = model or RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    freq = long_df.attrs.get("freq", "M")

    df = long_df[long_df["district"] == district].sort_values("period").reset_index(drop=True)
    feature_cols = _lag_feature_columns(df)
    df_model = df.dropna(subset=feature_cols + ["count"])

    if len(df_model) < test_size + 10:
        raise ValueError(f"Not enough complete rows for {district} to train/test a lag model.")

    X, y = df_model[feature_cols], df_model["count"]
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mape": mean_absolute_percentage_error(y_test, y_pred_test),
    }

    model.fit(X, y)  # refit on full history before forecasting forward

    history = df[["period", "count"]].copy()
    forecasts = []
    for _ in range(n_periods_ahead):
        next_period = history["period"].iloc[-1] + 1
        temp = pd.concat(
            [history, pd.DataFrame([{"period": next_period, "count": np.nan}])],
            ignore_index=True,
        )
        temp["district"] = district
        temp.attrs["freq"] = freq
        temp = make_lag_features(temp)
        x_next = temp.iloc[[-1]][feature_cols]

        if x_next.isna().any(axis=1).iloc[0]:
            break  # shouldn't happen once there's enough warm-up history

        pred = max(0, model.predict(x_next)[0])
        forecasts.append({"period": next_period, "forecast_count": round(pred)})
        history = pd.concat(
            [history, pd.DataFrame([{"period": next_period, "count": pred}])],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame(forecasts)
    if not forecast_df.empty:
        forecast_df["date"] = forecast_df["period"].apply(lambda p: p.to_timestamp())

    return {"model": model, "metrics": metrics, "forecast": forecast_df}


# --------------------------------------------------------------------------
# 2. Prophet
# --------------------------------------------------------------------------
def prophet_forecast(long_df: pd.DataFrame, district: str, n_periods_ahead: int = 6, test_size: int = 6):
    

    freq = long_df.attrs.get("freq", "M")
    df = long_df[long_df["district"] == district].sort_values("period").copy()
    df["ds"] = _to_timestamp_series(df)
    df["y"] = df["count"]
    prophet_df = df[["ds", "y"]].dropna()

    freq_str = "MS" if freq == "M" else "YS"

    # Backtest: fit on all but the last test_size periods, score on the held-out tail
    train_df = prophet_df.iloc[:-test_size]
    test_df = prophet_df.iloc[-test_size:]
    bt_model = Prophet(
        yearly_seasonality=(freq == "M"), weekly_seasonality=False, daily_seasonality=False
    )
    bt_model.fit(train_df)
    bt_future = bt_model.make_future_dataframe(periods=test_size, freq=freq_str)
    bt_forecast = bt_model.predict(bt_future).tail(test_size)
    metrics = {
        "mae": mean_absolute_error(test_df["y"], bt_forecast["yhat"]),
        "rmse": np.sqrt(mean_squared_error(test_df["y"], bt_forecast["yhat"])),
        "mape": mean_absolute_percentage_error(test_df["y"], bt_forecast["yhat"]),
    }

    # Final model: refit on full history, forecast forward
    model = Prophet(yearly_seasonality=(freq == "M"), weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=n_periods_ahead, freq=freq_str)
    forecast = model.predict(future)

    forecast_tail = forecast.tail(n_periods_ahead)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_tail = forecast_tail.rename(columns={"ds": "date", "yhat": "forecast_count"})
    forecast_tail["forecast_count"] = forecast_tail["forecast_count"].clip(lower=0).round()

    return {"model": model, "metrics": metrics, "forecast": forecast_tail}


# --------------------------------------------------------------------------
# 3. statsmodels — ETS (Exponential Smoothing) or SARIMA
# --------------------------------------------------------------------------
def statsmodels_forecast(
    long_df: pd.DataFrame,
    district: str,
    n_periods_ahead: int = 6,
    method: str = "ets",
    test_size: int = 6,
):
    

    freq = long_df.attrs.get("freq", "M")
    df = long_df[long_df["district"] == district].sort_values("period").copy()
    series = df.set_index(_to_timestamp_series(df))["count"]
    series = series.asfreq("MS" if freq == "M" else "YS")

    seasonal_periods = 12 if freq == "M" else None
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]

    def _fit_predict(train_series, horizon):
        if method == "ets":
            fitted = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal="add" if seasonal_periods else None,
                seasonal_periods=seasonal_periods,
            ).fit()
            return fitted, fitted.forecast(horizon)
        elif method == "sarima":
            seasonal_order = (1, 1, 1, 12) if freq == "M" else (0, 0, 0, 0)
            fitted = SARIMAX(
                train_series, order=(1, 1, 1), seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            return fitted, fitted.forecast(horizon)
        else:
            raise ValueError("method must be 'ets' or 'sarima'")

    _, bt_preds = _fit_predict(train, len(test))
    metrics = {
        "mae": mean_absolute_error(test, bt_preds),
        "rmse": np.sqrt(mean_squared_error(test, bt_preds)),
        "mape": mean_absolute_percentage_error(test, bt_preds),
    }

    model, preds = _fit_predict(series, n_periods_ahead)  # refit on full history
    forecast_df = pd.DataFrame({
        "date": preds.index,
        "forecast_count": preds.values.clip(min=0).round(),
    })

    return {"model": model, "metrics": metrics, "forecast": forecast_df}


# --------------------------------------------------------------------------
# 4. Run all three and line the results up for comparison
# --------------------------------------------------------------------------
def compare_models(
    long_df: pd.DataFrame,
    district: str,
    n_periods_ahead: int = 6,
    test_size: int = 6,
    sarima_or_ets: str = "ets",
):
    sk = sklearn_forecast(long_df, district, n_periods_ahead, test_size=test_size)
    pr = prophet_forecast(long_df, district, n_periods_ahead, test_size=test_size)
    sm = statsmodels_forecast(long_df, district, n_periods_ahead, method=sarima_or_ets, test_size=test_size)

    metrics_df = pd.DataFrame({
        "sklearn (RandomForest)": sk["metrics"],
        "Prophet": pr["metrics"],
        f"statsmodels ({sarima_or_ets.upper()})": sm["metrics"],
    }).T

    forecasts = {
        "sklearn (RandomForest)": sk["forecast"][["date", "forecast_count"]],
        "Prophet": pr["forecast"][["date", "forecast_count"]],
        f"statsmodels ({sarima_or_ets.upper()})": sm["forecast"][["date", "forecast_count"]],
    }

    return {"metrics": metrics_df, "forecasts": forecasts, "raw": {"sklearn": sk, "prophet": pr, "statsmodels": sm}}