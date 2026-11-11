"""
forecast.py — Demand forecasting for the AI-in-Operations dashboard.

Three models, increasing sophistication, all scored HONESTLY on a held-out window
(the last 28 days), never on their own training data:

    naive           -> tomorrow = today            (the lecturer's baseline)
    seasonal_naive  -> today = same weekday a week ago   (a fair, stronger baseline)
    lightgbm        -> gradient-boosted trees on calendar + price + event features

The single most important output is NOT the point forecast — it is sigma_d, the
standard deviation of the forecast's ERROR on the holdout. That number is what the
safety-stock formula consumes. A good forecast shrinks sigma_d, which shrinks the
buffer you must carry. This is the real link between "AI" and inventory savings.

All randomness is seeded. Features never peek at the future (lags/rolls are shifted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False

SEED = 42
HORIZON = 28  # days held out for honest testing


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum|e| / sum(y). Robust to zero days,
    which is why it is the primary metric here (MAPE explodes on the slow mover)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, computed only over non-zero actuals.
    Reported for the fast movers; meaningless where demand is mostly zero."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "WAPE": round(wape(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 2),
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 2),
        "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 2),
    }


# --------------------------------------------------------------------------- #
# Feature engineering  (no leakage: every lag/roll is shifted into the past)
# --------------------------------------------------------------------------- #
LAGS = [1, 7, 14, 28]
ROLLS = [7, 28]


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add calendar, lag, rolling and price/event features to a single-SKU frame.
    Expects columns: date, units, sell_price, snap_CA, has_event."""
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["date"])

    # calendar
    d["dayofweek"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["month"] = d["date"].dt.month
    d["day"] = d["date"].dt.day
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)

    # lags & rolling stats on past demand only
    for lag in LAGS:
        d[f"lag_{lag}"] = d["units"].shift(lag)
    for w in ROLLS:
        d[f"roll_mean_{w}"] = d["units"].shift(1).rolling(w).mean()
    d["roll_std_7"] = d["units"].shift(1).rolling(7).std()

    # price signal: relative discount vs recent typical price = promo proxy
    d["sell_price"] = d["sell_price"].ffill()
    d["price_roll_28"] = d["sell_price"].shift(1).rolling(28, min_periods=1).mean()
    d["price_rel"] = d["sell_price"] / d["price_roll_28"]

    feat_cols = (
        ["dayofweek", "is_weekend", "month", "day", "weekofyear",
         "snap_CA", "has_event", "sell_price", "price_rel"]
        + [f"lag_{l}" for l in LAGS]
        + [f"roll_mean_{w}" for w in ROLLS]
        + ["roll_std_7"]
    )
    return d, feat_cols


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    sku: str
    model: str
    holdout: pd.DataFrame          # date, y_true, y_pred
    residuals: np.ndarray
    sigma_d: float                 # std of holdout forecast error -> feeds safety stock
    demand_mean: float             # recent mean daily demand -> feeds ROP
    metrics: dict
    all_metrics: dict = field(default_factory=dict)  # every model, for comparison
    fitted_model: object = None
    feature_cols: list = field(default_factory=list)
    zero_day_pct: float = 0.0
    intermittent: bool = False


# --------------------------------------------------------------------------- #
# Individual models
# --------------------------------------------------------------------------- #
def _predict_naive(d: pd.DataFrame, test_idx) -> np.ndarray:
    return d["units"].shift(1).to_numpy()[test_idx]


def _predict_seasonal_naive(d: pd.DataFrame, test_idx, season=7) -> np.ndarray:
    return d["units"].shift(season).to_numpy()[test_idx]


def _predict_lightgbm(d: pd.DataFrame, feat_cols, train_idx, test_idx):
    model = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1,
    )
    Xtr, ytr = d.loc[train_idx, feat_cols], d.loc[train_idx, "units"]
    model.fit(Xtr, ytr)
    preds = model.predict(d.loc[test_idx, feat_cols])
    return np.clip(preds, 0, None), model  # demand can't be negative


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_forecast(
    df_sku: pd.DataFrame,
    model: str = "lightgbm",
    horizon: int = HORIZON,
    recent_window: int = 90,
) -> ForecastResult:
    """Fit baselines + the chosen model, score all on the last `horizon` days,
    and return the pieces the rest of the dashboard needs (especially sigma_d)."""
    sku = str(df_sku["sku"].iloc[0]) if "sku" in df_sku.columns else "SKU"
    d2, feat_cols = make_features(df_sku)
    # drop warm-up rows where lags/rolls are undefined
    d2 = d2.dropna(subset=feat_cols).reset_index(drop=True)

    n = len(d2)
    if n <= horizon + 10:
        horizon = max(7, n // 5)
    split = n - horizon
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)

    y_true = d2["units"].to_numpy()[test_idx]

    # always compute baselines for the comparison table
    all_metrics = {}
    preds_naive = _predict_naive(d2, test_idx)
    preds_snaive = _predict_seasonal_naive(d2, test_idx)
    all_metrics["naive"] = _metrics(y_true, preds_naive)
    all_metrics["seasonal_naive"] = _metrics(y_true, preds_snaive)

    preds_by_model = {"naive": preds_naive, "seasonal_naive": preds_snaive}
    fitted = None
    if _HAS_LGBM:
        preds_lgbm, fitted = _predict_lightgbm(d2, feat_cols, train_idx, test_idx)
        all_metrics["lightgbm"] = _metrics(y_true, preds_lgbm)
        preds_by_model["lightgbm"] = preds_lgbm

    if model == "best":
        # Honest auto-select: use whichever model has the lowest holdout WAPE.
        # This is how the dashboard avoids using ML where a baseline is better.
        model = min(preds_by_model, key=lambda m: all_metrics[m]["WAPE"])
    elif model not in preds_by_model:
        model = "naive"
    chosen = preds_by_model[model]

    residuals = y_true - chosen
    sigma_d = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    demand_mean = float(d2["units"].to_numpy()[-recent_window:].mean())
    zero_pct = float((d2["units"] == 0).mean() * 100)

    holdout = pd.DataFrame({
        "date": d2["date"].to_numpy()[test_idx],
        "y_true": y_true,
        "y_pred": chosen,
    })

    return ForecastResult(
        sku=sku, model=model, holdout=holdout, residuals=residuals,
        sigma_d=sigma_d, demand_mean=demand_mean,
        metrics=all_metrics.get(model, _metrics(y_true, chosen)),
        all_metrics=all_metrics, fitted_model=fitted, feature_cols=feat_cols,
        zero_day_pct=zero_pct, intermittent=zero_pct > 60,
    )


if __name__ == "__main__":
    # Quick self-check on the real filtered data.
    df = pd.read_csv("m5_filtered.csv", parse_dates=["date"])
    for sku in df["sku"].unique():
        r = run_forecast(df[df.sku == sku], model="lightgbm")
        comp = " | ".join(f"{k}:WAPE={v['WAPE']}" for k, v in r.all_metrics.items())
        flag = "  <-- INTERMITTENT, formula unreliable" if r.intermittent else ""
        print(f"{sku:18s} {comp}  || sigma_d={r.sigma_d:6.2f} Dbar={r.demand_mean:6.2f}{flag}")
