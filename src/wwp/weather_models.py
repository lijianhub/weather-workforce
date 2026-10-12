"""Weather risk models for tomorrow's working hours (07:00-17:00).

Ladder of models, each must beat the one before to earn its complexity:
  1. climatology        - "how often does it rain on this month's work days?"
  2. persistence        - "tomorrow is like today"
  3. logistic (history) - linear model on today's observations
  4. LightGBM (history) - trees on today's observations
  5. NWP calibrated     - the professional day-ahead forecast, mapped to a probability
  6. MOS                - forecast + local signals, corrected for this orchard

What the planner needs is not "mm of rain" but "how many working hours will be wet",
so besides P(rain >= 1 mm) we predict the wet-hours distribution with three stacked
yes/no models: P(>= 1 wet hour), P(>= 4) (= washout), P(>= 7).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_absolute_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config as C
from .features import HISTORY_COLS, MOS_COLS, NWP_COLS

warnings.filterwarnings("ignore", module="lightgbm")

# name -> (target column, raw forecast field used by the one-feature baseline)
TARGETS = {
    "rain": ("y_rain", "nwp_rain_mm"),           # >= 1 mm in work hours (reported)
    "wet1": ("y_wet1", "nwp_rain_hours"),        # >= 1 wet work hour
    "washout": ("y_washout", "nwp_rain_hours"),  # >= 4 wet work hours
    "wet7": ("y_wet7", "nwp_rain_hours"),        # >= 7 wet work hours
}
REPORTED = ["rain", "washout"]
RAIN_LIKE = ["today_rain_mm", "rain_yesterday_mm", "rain_prev_7d_mm",
             "nwp_rain_mm", "nwp_rain_max_hourly"]


def _log_rain(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in RAIN_LIKE:
        if c in X:
            X[c] = np.log1p(X[c])
    return X


def _linear(model):
    return make_pipeline(FunctionTransformer(_log_rain), StandardScaler(), model)


def split(df: pd.DataFrame):
    train = df[df.index <= C.TRAIN_END]
    valid = df[(df.index > C.TRAIN_END) & (df.index <= C.VALID_END)]
    test = df[df.index > C.VALID_END]
    return train, valid, test


def fit_history_models(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """LightGBM on observations only: the fallback when the forecast feed is down."""
    m = {}
    for name, (target, _) in TARGETS.items():
        gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        gbm.fit(train[HISTORY_COLS], train[target],
                eval_set=[(valid[HISTORY_COLS], valid[target])],
                callbacks=[lgb.early_stopping(100, verbose=False)])
        m[f"{name}_lgb"] = gbm
    mm = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                           n_estimators=2000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1)
    mm.fit(train[HISTORY_COLS], train["obs_rain_mm"],
           eval_set=[(valid[HISTORY_COLS], valid["obs_rain_mm"])],
           callbacks=[lgb.early_stopping(100, verbose=False)])
    m["mm_lgb"] = mm
    return m


def fit_forecast_models(nwp_rows: pd.DataFrame) -> dict:
    """Forecast-based models. Small and linear: they retrain cheaply on recent data."""
    m = {}
    for name, (target, raw_col) in TARGETS.items():
        m[f"{name}_nwp"] = _linear(LogisticRegression(max_iter=2000)).fit(
            nwp_rows[[raw_col]], nwp_rows[target])
        m[f"{name}_mos"] = _linear(LogisticRegression(max_iter=2000, C=0.3)).fit(
            nwp_rows[MOS_COLS], nwp_rows[target])
    m["mm_mos"] = _linear(PoissonRegressor(alpha=0.1, max_iter=2000)).fit(
        nwp_rows[MOS_COLS], nwp_rows["obs_rain_mm"])
    return m


class WeatherRiskModel:
    """Serving wrapper: MOS when the forecast feed is available, else history model."""

    def __init__(self, models: dict):
        self.m = models

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        has_nwp = X[NWP_COLS].notna().all(axis=1).to_numpy()
        out = pd.DataFrame(index=X.index)
        for name in TARGETS:
            p = self.m[f"{name}_lgb"].predict_proba(X[HISTORY_COLS])[:, 1]
            if has_nwp.any():
                p[has_nwp] = self.m[f"{name}_mos"].predict_proba(X.loc[has_nwp, MOS_COLS])[:, 1]
            out[f"p_{name}"] = p
        # The wet-hour probabilities must be nested: P(>=7) <= P(>=4) <= P(>=1).
        out["p_washout"] = np.minimum(out["p_washout"], out["p_wet1"])
        out["p_wet7"] = np.minimum(out["p_wet7"], out["p_washout"])
        mm = self.m["mm_lgb"].predict(X[HISTORY_COLS])
        if has_nwp.any():
            mm[has_nwp] = self.m["mm_mos"].predict(X.loc[has_nwp, MOS_COLS])
        out["expected_mm"] = np.clip(mm, 0, None)
        out["nwp_rain_mm"] = X["nwp_rain_mm"].to_numpy()
        out["nwp_rain_hours"] = X["nwp_rain_hours"].to_numpy()
        out["source"] = np.where(has_nwp, "MOS (forecast + local)", "history-only fallback")
        return out


def walk_forward(df: pd.DataFrame, history_models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Predict each year with forecast models retrained on all earlier years only,
    as a production system retrained every winter would."""
    nwp_rows = df.dropna(subset=NWP_COLS)
    parts = []
    for year, chunk in rows.groupby(rows.index.year):
        past = nwp_rows[nwp_rows.index < f"{year}-01-01"]
        model = WeatherRiskModel({**history_models, **fit_forecast_models(past)})
        parts.append(model.predict(chunk))
    return pd.concat(parts)


def _clf_scores(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}


def train_and_evaluate(df: pd.DataFrame):
    """Fit everything with a time split and score each rung of the ladder on 2025-26."""
    train, valid, test = split(df)
    valid_nwp = valid.dropna(subset=NWP_COLS)
    test = test.dropna(subset=NWP_COLS)     # same rows for every model -> fair comparison
    hist = fit_history_models(train, valid)
    fc = fit_forecast_models(valid_nwp)     # MOS trained on 2024 only for this table
    metrics = {}

    for name in REPORTED:
        target, raw_col = TARGETS[name]
        P = {}
        clim = train.groupby(train.index.month)[target].mean()
        P["climatology"] = test.index.month.map(clim).to_numpy()
        rate = train.groupby(train["today_rain_mm"] >= C.RAIN_MM)[target].mean()
        P["persistence"] = (test["today_rain_mm"] >= C.RAIN_MM).map(rate).to_numpy()
        lin = _linear(LogisticRegression(max_iter=2000, C=0.5)).fit(train[HISTORY_COLS],
                                                                     train[target])
        P["logistic (history)"] = lin.predict_proba(test[HISTORY_COLS])[:, 1]
        P["LightGBM (history)"] = hist[f"{name}_lgb"].predict_proba(test[HISTORY_COLS])[:, 1]
        P["NWP forecast (calibrated)"] = fc[f"{name}_nwp"].predict_proba(test[[raw_col]])[:, 1]
        P["MOS (forecast + local)"] = fc[f"{name}_mos"].predict_proba(test[MOS_COLS])[:, 1]
        metrics[name] = {k: _clf_scores(test[target], v) for k, v in P.items()}
        metrics[name]["_base_rate_test"] = float(test[target].mean())

    y = test["obs_rain_mm"]
    clim_mm = train.groupby(train.index.month)["obs_rain_mm"].mean()
    metrics["rain_mm_mae"] = {
        "climatology": mean_absolute_error(y, test.index.month.map(clim_mm)),
        "LightGBM (history)": mean_absolute_error(y, hist["mm_lgb"].predict(test[HISTORY_COLS])),
        "NWP forecast (raw)": mean_absolute_error(y, test["nwp_rain_mm"]),
        "MOS (forecast + local)": mean_absolute_error(y, fc["mm_mos"].predict(test[MOS_COLS])),
    }
    metrics["split"] = {
        "history_train": [str(train.index.min().date()), str(train.index.max().date())],
        "mos_train": [str(valid_nwp.index.min().date()), str(valid_nwp.index.max().date())],
        "test": [str(test.index.min().date()), str(test.index.max().date())],
        "n_test_days": int(len(test))}
    # Serving model: forecast models refit on every forecast day we have.
    serving = WeatherRiskModel({**hist, **fit_forecast_models(df.dropna(subset=NWP_COLS))})
    return serving, hist, metrics, test
