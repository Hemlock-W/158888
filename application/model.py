"""
Evaluation standard:
  - expanding-window walk-forward CV (not just one holdout split)
  - optional log1p target transform (use_log)
  - hyperparameter tuning (RandomizedSearchCV for sklearn; small grid
    searches over each library's own knobs for Prophet/statsmodels,
    scored via the same walk-forward CV)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

from feature_engineering import build_features, get_feature_cols

def _to_timestamp_series(df:pd.DataFrame) -> pd.Series:
    return df["period"].apply(lambda p: p.to_timestamp())


def _fold_metrics_from_preds(y_true, y_pred, fold_idx):
    return {
        "fold": fold_idx,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def _aggregate_fold_metrics(fold_metrics:list[dict]) -> dict:
    return {
        "mae": np.mean([m["mae"] for m in fold_metrics]),
        "rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "mape": np.mean([m["mape"] for m in fold_metrics]),
    }


# --------------------------------------------------------------------------
# Shared expanding-window walk-forward CV — used by Prophet and statsmodels
# --------------------------------------------------------------------------
def walk_forward_evaluate_series(
    series:pd.Series,
    fit_predict_fn,
    n_splits:int  = 5,
    test_size:int = 6,
    min_train:int = 24,
):
    n = len(series)
    if n < min_train + test_size:
        raise ValueError(
            f"Not enough data for walk-forward CV: have {n} rows, need at "
            f"least {min_train + test_size}."
        )

    fold_metrics = []
    step = max(1, (n - min_train - test_size) // n_splits)

    for i in range(n_splits):
        test_end = n - i * step
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train:
            break

        train_series = series.iloc[:train_end]
        test_series = series.iloc[test_start:test_end]

        y_pred = np.maximum(np.asarray(fit_predict_fn(train_series, len(test_series))), 0)
        fold_metrics.append(_fold_metrics_from_preds(test_series.values, y_pred, i))

    if not fold_metrics:
        raise ValueError("No valid folds — series too short for the requested n_splits/test_size.")

    return _aggregate_fold_metrics(fold_metrics), fold_metrics


# --------------------------------------------------------------------------
# 1. Scikit-learn 
# --------------------------------------------------------------------------
def walk_forward_evaluate(
    df:pd.DataFrame,
    feature_cols:list,
    model,
    n_splits:int  = 5,
    test_size:int = 10,
    use_log:bool  = True,
):
    """Expanding-window walk-forward CV for an sklearn-style model (X, y)."""
    df_clean = df.dropna(subset=feature_cols + ["count"]).reset_index(drop=True)
    n = len(df_clean)

    if n < test_size + 24:
        raise ValueError("Not enough data for walk-forward CV.")

    fold_metrics = []
    min_train = 24
    step = max(1, (n - min_train - test_size) // n_splits)

    for i in range(n_splits):
        test_end = n - i * step
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train:
            break

        X_train = df_clean[feature_cols].iloc[:train_end]
        y_train = df_clean["count"].iloc[:train_end]
        X_test = df_clean[feature_cols].iloc[test_start:test_end]
        y_test = df_clean["count"].iloc[test_start:test_end]

        if use_log:
            model.fit(X_train, np.log1p(y_train))
            y_pred = np.expm1(model.predict(X_test))
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        y_pred = np.maximum(y_pred, 0)
        fold_metrics.append(_fold_metrics_from_preds(y_test, y_pred, i))

    return _aggregate_fold_metrics(fold_metrics), fold_metrics


def get_tuned_model(X_train, y_train, use_log=True):
    y_train_t = np.log1p(y_train) if use_log else y_train

    param_distributions = {
        "n_estimators": randint(100, 500), #400
        "max_depth": randint(3, 8), #6
        "learning_rate": uniform(0.01, 0.2),    #0.2
        "subsample": uniform(0.6, 0.4), #0.6
        "min_samples_split": randint(2, 20),    #9
        "min_samples_leaf": randint(1, 10), #16
    }

    base = GradientBoostingRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        base, param_distributions, n_iter=60, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train_t)
    return search.best_estimator_, search.best_params_


def sklearn_forecast(
    long_df:pd.DataFrame,
    district:str,
    n_periods_ahead:int = 6,
    model               = None,
    test_size:int       = 10,
    use_log:bool        = True,
    tune:bool           = True,
):
    freq = long_df.attrs.get("freq", "M")
    df = long_df[long_df["district"] == district].sort_values("period").reset_index(drop=True)

    if not isinstance(df["period"].dtype, pd.PeriodDtype):
        df["period"] = pd.PeriodIndex(df["period"], freq=freq)

    if "time_idx" not in df.columns:
        df = build_features(df, freq=freq)

    feature_cols = get_feature_cols(df)
    df_model = df.dropna(subset=feature_cols + ["count"]).reset_index(drop=True)

    if len(df_model) < test_size + 18:
        raise ValueError(f"Not enough rows for {district}.")

    X, y = df_model[feature_cols], df_model["count"]

    X, y = df_model[feature_cols], df_model["count"]
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Tune (using only X_train, never the final holdout), then reuse
    # the resulting hyperparameters for walk-forward CV  
    # Nested CV removes that bias but is far more expensive
    if tune:
        best_model, best_params = get_tuned_model(X_train, y_train, use_log=use_log)
        cv_model = clone(best_model)
    else:
        best_model = model
        best_model.fit(X_train, np.log1p(y_train) if use_log else y_train)
        best_params = None
        cv_model = clone(model)

    agg_metrics, fold_metrics = walk_forward_evaluate(
        df_model, feature_cols, cv_model, n_splits=5, test_size=test_size, use_log=use_log
    )

    y_pred_test = best_model.predict(X_test)
    if use_log:
        y_pred_test = np.expm1(y_pred_test)
    y_pred_test = np.maximum(y_pred_test, 0)

    holdout_metrics = {
        "mae": mean_absolute_error(y_test, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mape": mean_absolute_percentage_error(y_test, y_pred_test),
    }

    holdout_df = pd.DataFrame({
        "period": df_model["period"].iloc[-test_size:].values,
        "actual": y_test.values,
        "predicted": y_pred_test,
    })
    holdout_df["date"] = holdout_df["period"].apply(lambda p: p.to_timestamp())

    best_model.fit(X, np.log1p(y) if use_log else y)

    history = df[["period", "count"]].copy()
    forecasts = []
    for _ in range(n_periods_ahead):
        next_period = history["period"].iloc[-1] + 1
        temp = pd.concat(
            [history, pd.DataFrame([{"period": next_period, "count": np.nan}])], ignore_index=True
        )
        temp = build_features(temp, freq=freq)
        x_next = temp[feature_cols].iloc[[-1]]

        if x_next.isna().any(axis=1).iloc[0]:
            break

        pred = best_model.predict(x_next)[0]
        pred = np.expm1(pred) if use_log else pred
        pred = max(0, pred)

        forecasts.append({"period": next_period, "forecast_count": round(pred)})
        history = pd.concat(
            [history, pd.DataFrame([{"period": next_period, "count": pred}])], ignore_index=True
        )

    forecast_df = pd.DataFrame(forecasts)
    if not forecast_df.empty:
        forecast_df["date"] = forecast_df["period"].apply(lambda p: p.to_timestamp())

    return {
        "model": best_model,
        "metrics_holdout": holdout_metrics,
        "metrics_cv": agg_metrics,
        "fold_metrics": fold_metrics,
        "forecast": forecast_df,
        "best_params": best_params,
        "holdout": holdout_df,
    }


# --------------------------------------------------------------------------
# 2. Prophet
# --------------------------------------------------------------------------
PROPHET_PARAM_GRID = [
    {"changepoint_prior_scale": 0.25, "seasonality_prior_scale": 30.0, "seasonality_mode": "additive"},
]


def _prophet_fit_predict(train_series:pd.Series, horizon:int, freq_str:str, params:dict, use_log:bool):
    train_df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
    if use_log:
        train_df["y"] = np.log1p(train_df["y"])

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, **params
    )
    model.fit(train_df)
    future = model.make_future_dataframe(periods=horizon, freq=freq_str)
    preds = model.predict(future)["yhat"].tail(horizon).values
    return np.expm1(preds) if use_log else preds


def get_tuned_prophet_params(series, freq_str, use_log, n_splits = 3, test_size = 6):
    """Small grid search over Prophet hyperparameters, scored via walk-forward CV."""
    best_params, best_score = None, np.inf
    for params in PROPHET_PARAM_GRID:
        fit_predict_fn = lambda train_s, h, p=params: _prophet_fit_predict(train_s, h, freq_str, p, use_log)
        try:
            agg, _ = walk_forward_evaluate_series(series, fit_predict_fn, n_splits=n_splits, test_size=test_size)
        except Exception:
            continue
        if agg["mae"] < best_score:
            best_score, best_params = agg["mae"], params
    return best_params or PROPHET_PARAM_GRID[0]


def prophet_forecast(
    long_df:pd.DataFrame,
    district:str,
    n_periods_ahead:int = 6,
    test_size:int       = 10,
    use_log:bool        = True,
    tune:bool           = True,
    n_splits:int        = 5,
):
    freq = long_df.attrs.get("freq", "M")
    freq_str = "MS" if freq == "M" else "YS"

    df = long_df[long_df["district"] == district].sort_values("period").copy()
    df["ds"] = _to_timestamp_series(df)
    series = df.set_index("ds")["count"].dropna()

    if len(series) < test_size + 24:
        raise ValueError(f"Not enough rows for {district}.")

    best_params = (
        get_tuned_prophet_params(series, freq_str, use_log, n_splits=3, test_size=test_size)
        if tune else PROPHET_PARAM_GRID[0]
    )

    fit_predict_fn = lambda train_s, h: _prophet_fit_predict(train_s, h, freq_str, best_params, use_log)
    agg_metrics, fold_metrics = walk_forward_evaluate_series(
        series, fit_predict_fn, n_splits=n_splits, test_size=test_size
    )

    train_series, test_series = series.iloc[:-test_size], series.iloc[-test_size:]
    y_pred_test = np.maximum(fit_predict_fn(train_series, len(test_series)), 0)
    holdout_metrics = {
        "mae": mean_absolute_error(test_series, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(test_series, y_pred_test)),
        "mape": mean_absolute_percentage_error(test_series, y_pred_test),
    }

    holdout_df = pd.DataFrame({
        "date": test_series.index,
        "actual": test_series.values,
        "predicted": y_pred_test,
    })

    full_df = pd.DataFrame({"ds": series.index, "y": series.values})
    if use_log:
        full_df["y"] = np.log1p(full_df["y"])
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, **best_params)
    model.fit(full_df)
    future = model.make_future_dataframe(periods=n_periods_ahead, freq=freq_str)
    forecast = model.predict(future)

    forecast_tail = forecast.tail(n_periods_ahead)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    if use_log:
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            forecast_tail[col] = np.expm1(forecast_tail[col])
    forecast_tail = forecast_tail.rename(columns={"ds": "date", "yhat": "forecast_count"})
    forecast_tail["forecast_count"] = forecast_tail["forecast_count"].clip(lower=0).round()

    return {
        "model": model,
        "metrics_holdout": holdout_metrics,
        "metrics_cv": agg_metrics,
        "fold_metrics": fold_metrics,
        "forecast": forecast_tail,
        "best_params": best_params,
        "holdout": holdout_df,
    }


# --------------------------------------------------------------------------
# 3. Statsmodels 
# --------------------------------------------------------------------------
ETS_PARAM_GRID = [
    {"trend": None, "damped_trend": True, "seasonal": "add"},
]

SARIMA_PARAM_GRID = [
    {"order": (1, 1, 0), "seasonal_order": (0, 1, 1, 12)},
]


def _ets_fit_predict(train_series, horizon, params, seasonal_periods, use_log):
    y = np.log1p(train_series) if use_log else train_series
    fitted = ExponentialSmoothing(
        y,
        trend=params.get("trend"),
        damped_trend=params.get("damped_trend", False) if params.get("trend") else False,
        seasonal=params.get("seasonal") if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
    ).fit()
    preds = fitted.forecast(horizon)
    return np.expm1(preds) if use_log else preds


def _sarima_fit_predict(train_series, horizon, params, use_log):
    y = np.log1p(train_series) if use_log else train_series
    fitted = SARIMAX(
        y, order=params["order"], seasonal_order=params["seasonal_order"],
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    preds = fitted.forecast(horizon)
    return np.expm1(preds) if use_log else preds


def get_tuned_statsmodels_params(series, method, seasonal_periods, use_log, n_splits=3, test_size=6):
    grid = ETS_PARAM_GRID if method == "ets" else SARIMA_PARAM_GRID
    best_params, best_score = None, np.inf

    for params in grid:
        if method == "ets":
            fit_predict_fn = lambda train_s, h, p=params: _ets_fit_predict(train_s, h, p, seasonal_periods, use_log)
        else:
            fit_predict_fn = lambda train_s, h, p=params: _sarima_fit_predict(train_s, h, p, use_log)
        try:
            agg, _ = walk_forward_evaluate_series(series, fit_predict_fn, n_splits=n_splits, test_size=test_size)
        except Exception:
            continue
        if agg["mae"] < best_score:
            best_score, best_params = agg["mae"], params

    return best_params or grid[0]


def statsmodels_forecast(
    long_df:pd.DataFrame,
    district:str,
    n_periods_ahead:int = 6,
    method:str          = "ets",
    test_size:int       = 10,
    use_log:bool        = True,
    tune:bool           = True,
    n_splits:int        = 5,
):
    freq = long_df.attrs.get("freq", "M")
    df = long_df[long_df["district"] == district].sort_values("period").copy()
    series = df.set_index(_to_timestamp_series(df))["count"]
    series = series.asfreq("MS" if freq == "M" else "YS").dropna()

    if len(series) < test_size + 24:
        raise ValueError(f"Not enough rows for {district}.")

    seasonal_periods = 12 if freq == "M" else None

    best_params = (
        get_tuned_statsmodels_params(series, method, seasonal_periods, use_log, n_splits=3, test_size=test_size)
        if tune else (ETS_PARAM_GRID if method == "ets" else SARIMA_PARAM_GRID)[0]
    )

    if method == "ets":
        fit_predict_fn = lambda train_s, h: _ets_fit_predict(train_s, h, best_params, seasonal_periods, use_log)
    elif method == "sarima":
        fit_predict_fn = lambda train_s, h: _sarima_fit_predict(train_s, h, best_params, use_log)
    else:
        raise ValueError("method must be 'ets' or 'sarima'")

    agg_metrics, fold_metrics = walk_forward_evaluate_series(
        series, fit_predict_fn, n_splits=n_splits, test_size=test_size
    )

    train_series, test_series = series.iloc[:-test_size], series.iloc[-test_size:]
    y_pred_test = np.maximum(fit_predict_fn(train_series, len(test_series)), 0)
    holdout_metrics = {
        "mae": mean_absolute_error(test_series, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(test_series, y_pred_test)),
        "mape": mean_absolute_percentage_error(test_series, y_pred_test),
    }

    holdout_df = pd.DataFrame({
        "date": test_series.index,
        "actual": test_series.values,
        "predicted": y_pred_test,
    })

    y_full = np.log1p(series) if use_log else series
    if method == "ets":
        model = ExponentialSmoothing(
            y_full,
            trend=best_params.get("trend"),
            damped_trend=best_params.get("damped_trend", False) if best_params.get("trend") else False,
            seasonal=best_params.get("seasonal") if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
        ).fit()
    else:
        model = SARIMAX(
            y_full, order=best_params["order"], seasonal_order=best_params["seasonal_order"],
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

    preds = model.forecast(n_periods_ahead)
    if use_log:
        preds = np.expm1(preds)
    preds = np.maximum(preds, 0)

    forecast_df = pd.DataFrame({"date": preds.index, "forecast_count": np.round(preds.values)})

    return {
        "model": model,
        "metrics_holdout": holdout_metrics,
        "metrics_cv": agg_metrics,
        "fold_metrics": fold_metrics,
        "forecast": forecast_df,
        "best_params": best_params,
        "holdout": holdout_df,
    }


# --------------------------------------------------------------------------
# 4. Run all three and line the results up for comparison
# --------------------------------------------------------------------------
def compare_models(
    long_df:pd.DataFrame,
    district:str,
    n_periods_ahead:int = 4,
    test_size:int       = 10,
    sarima_or_ets:str   = "ets",
    use_log:bool        = True,
    tune:bool           = True,
    n_splits:int        = 5,
):
    sk = sklearn_forecast(long_df, district, n_periods_ahead, test_size=test_size, use_log=use_log, tune=tune)
    pr = prophet_forecast(long_df, district, n_periods_ahead, test_size=test_size, use_log=use_log, tune=tune, n_splits=n_splits)
    sm = statsmodels_forecast(long_df, district, n_periods_ahead, method=sarima_or_ets, test_size=test_size, use_log=use_log, tune=tune, n_splits=n_splits)

    metrics_holdout = pd.DataFrame({
        "sklearn (GradientBoosting)": sk["metrics_holdout"],
        "Prophet": pr["metrics_holdout"],
        f"statsmodels ({sarima_or_ets.upper()})": sm["metrics_holdout"],
    }).T

    metrics_cv = pd.DataFrame({
        "sklearn (GradientBoosting)": sk["metrics_cv"],
        "Prophet": pr["metrics_cv"],
        f"statsmodels ({sarima_or_ets.upper()})": sm["metrics_cv"],
    }).T

    forecasts = {
        "sklearn (GradientBoosting)": sk["forecast"][["date", "forecast_count"]],
        "Prophet": pr["forecast"][["date", "forecast_count"]],
        f"statsmodels ({sarima_or_ets.upper()})": sm["forecast"][["date", "forecast_count"]],
    }

    return {
        "metrics_holdout": metrics_holdout,
        "metrics_cv": metrics_cv,
        "forecasts": forecasts,
        "raw": {"sklearn": sk, "prophet": pr, "statsmodels": sm},
    }