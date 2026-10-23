"""Forecasters (GRU / LSTM from the notebook, plus a gradient-boosting baseline),
back-testing, and empirical prediction intervals."""
from __future__ import annotations
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES, LOOKBACK, ROOT, TrafficData

MODEL_DIR = ROOT / "models"
H = (1, 2)


# ---------------------------------------------------------------- scaler
def load_scale() -> tuple[float, float]:
    """Min/max of the notebook's MinMaxScaler (fitted on the training split)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = joblib.load(MODEL_DIR / "traffic_scaler.pkl")
        return float(sc.data_min_[0]), float(sc.data_max_[0])
    except Exception:
        return 0.0, 7280.0


# ---------------------------------------------------------------- forecasters
class NNForecaster:
    """Keras GRU/LSTM: 24 scaled hourly values in, next hour out.  The 2-hour
    forecast feeds the 1-hour prediction back in (recursive)."""

    def __init__(self, name: str, model, dmin: float, dmax: float):
        self.name, self.model, self.dmin, self.dmax = name, model, dmin, dmax

    def _step(self, vol: np.ndarray) -> np.ndarray:
        x = ((vol - self.dmin) / (self.dmax - self.dmin)).astype("float32")[..., None]
        y = np.asarray(self.model.predict(x, batch_size=1024, verbose=0)).reshape(-1)
        return np.clip(y * (self.dmax - self.dmin) + self.dmin, 0, None)

    def predict_pair(self, data: TrafficData, times):
        w = data.windows(times)
        p1 = self._step(w)
        p2 = self._step(np.concatenate([w[:, 1:], p1[:, None]], axis=1))
        return p1, p2


class EnsembleForecaster:
    def __init__(self, name: str, members: list):
        self.name, self.members = name, members

    def predict_pair(self, data, times):
        out = [m.predict_pair(data, times) for m in self.members]
        return np.mean([o[0] for o in out], axis=0), np.mean([o[1] for o in out], axis=0)


class GBForecaster:
    """Gradient-boosting baseline: one model per horizon, trained on the notebook's
    training period using current/lagged volume, rolling means, calendar and weather."""

    def __init__(self, name: str, m1, m2):
        self.name, self.m1, self.m2 = name, m1, m2

    @staticmethod
    def train(data: TrafficData, name="Gradient Boosting"):
        from sklearn.ensemble import GradientBoostingRegressor
        f = data.features
        y1, y2 = data.full.shift(-1), data.full.shift(-2)
        ok = f[FEATURES].notna().all(axis=1) & y1.notna() & y2.notna() & (f.index < data.split_ts)
        X = f.loc[ok, FEATURES].to_numpy()
        m1 = GradientBoostingRegressor(random_state=42).fit(X, y1[ok].to_numpy())
        m2 = GradientBoostingRegressor(random_state=42).fit(X, y2[ok].to_numpy())
        return GBForecaster(name, m1, m2)

    def predict_pair(self, data, times):
        X = data.feature_matrix(times)
        return np.clip(self.m1.predict(X), 0, None), np.clip(self.m2.predict(X), 0, None)


def _load_keras(path: Path):
    try:
        import keras  # Keras 3 (ships with TensorFlow >= 2.16)
        return keras.models.load_model(path, compile=False)
    except Exception:
        from tensorflow import keras as tfk  # older installs
        return tfk.models.load_model(path, compile=False)


def build_forecasters(data: TrafficData):
    """Returns ({name: forecaster}, [notes]).  Falls back to gradient boosting if
    TensorFlow/Keras or the model files are unavailable."""
    notes, fcs = [], {}
    dmin, dmax = load_scale()
    nn = []
    for label, fname in (("GRU", "traffic_gru.keras"), ("LSTM", "traffic_lstm.keras")):
        try:
            nn.append(NNForecaster(label, _load_keras(MODEL_DIR / fname), dmin, dmax))
        except Exception as e:  # noqa: BLE001
            notes.append(f"Could not load {fname} ({type(e).__name__}). Install TensorFlow to use the {label} model.")
    for m in nn:
        fcs[m.name] = m
    if len(nn) == 2:
        fcs["Ensemble (GRU + LSTM)"] = EnsembleForecaster("Ensemble (GRU + LSTM)", nn)

    gb_path = MODEL_DIR / "gb_baseline.joblib"
    gb = None
    try:
        import sklearn
        if gb_path.exists():
            try:  # reuse the cached baseline only if it was trained with this scikit-learn version
                blob = joblib.load(gb_path)
                if blob.get("sklearn") == sklearn.__version__:
                    gb = GBForecaster("Gradient Boosting", blob["m1"], blob["m2"])
            except Exception:
                gb = None
        if gb is None:
            gb = GBForecaster.train(data)
            try:
                joblib.dump({"sklearn": sklearn.__version__, "m1": gb.m1, "m2": gb.m2}, gb_path)
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        notes.append(f"Gradient-boosting baseline unavailable ({type(e).__name__}).")
    if gb is not None:
        fcs["Gradient Boosting"] = gb
    return fcs, notes


# ---------------------------------------------------------------- back-test
def backtest(data: TrafficData, fcs: dict) -> pd.DataFrame:
    """Forecast every valid hour in the notebook's test period and compare with what happened.
    Long format: time, model, h, pred, actual (+ same-hour-last-week baseline)."""
    times = data.valid_asof[data.valid_asof >= data.split_ts]
    rows = []
    for h in H:
        tgt = times + pd.Timedelta(hours=h)
        actual = data.full.reindex(tgt).to_numpy()
        week = data.full.reindex(tgt - pd.Timedelta(days=7)).to_numpy()
        keep = ~np.isnan(actual)
        rows.append(pd.DataFrame({"time": times[keep], "model": "Same hour last week", "h": h,
                                  "pred": week[keep], "actual": actual[keep]}))
    for name, fc in fcs.items():
        p = fc.predict_pair(data, times)
        for h, ph in zip(H, p):
            actual = data.full.reindex(times + pd.Timedelta(hours=h)).to_numpy()
            keep = ~np.isnan(actual)
            rows.append(pd.DataFrame({"time": times[keep], "model": name, "h": h,
                                      "pred": ph[keep], "actual": actual[keep]}))
    bt = pd.concat(rows, ignore_index=True)
    bt["target_hour"] = (bt["time"] + pd.to_timedelta(bt["h"], unit="h")).dt.hour
    bt["resid"] = bt["actual"] - bt["pred"]
    return bt


def metrics_table(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE per model and horizon, on rows where every model (and the weekly baseline) has a value."""
    out = []
    for h in H:
        wide = bt[bt.h == h].pivot_table(index="time", columns="model", values="pred")
        act = bt[bt.h == h].groupby("time")["actual"].first()
        wide = wide.dropna()
        for m in wide.columns:
            e = act.loc[wide.index] - wide[m]
            out.append({"Model": m, "Horizon": f"{h}h", "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())),
                        "Hours tested": len(e)})
    t = pd.DataFrame(out)
    return t.sort_values(["Horizon", "MAE"]).reset_index(drop=True)


def residual_bands(bt: pd.DataFrame, level: float = 0.90) -> dict:
    """Empirical prediction-interval offsets (actual - prediction) by model, horizon and hour of day."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    bands = {}
    for (m, h), g in bt[bt.model != "Same hour last week"].groupby(["model", "h"]):
        overall = g["resid"].quantile([lo_q, hi_q]).to_numpy()
        per = g.groupby("target_hour")["resid"].agg(lo=lambda x: x.quantile(lo_q), hi=lambda x: x.quantile(hi_q),
                                                    n="count").reindex(range(24))
        weak = per["n"].fillna(0) < 30
        per.loc[weak, ["lo", "hi"]] = overall
        bands[(m, h)] = per[["lo", "hi"]]
    return bands


def interval(bands: dict, model: str, h: int, target_hour: int, pred: float):
    """(lo, hi) volume for a forecast, from the empirical residuals of that hour of day."""
    b = bands[(model, h)].loc[target_hour]
    return max(0.0, pred + b["lo"]), max(0.0, pred + b["hi"])


def exceed_probability(bt: pd.DataFrame, model: str, h: int, target_hour: int, pred: float, threshold: float) -> float:
    """Share of back-tested errors (for that model/horizon/hour of day) that would put actual volume at/above threshold."""
    g = bt[(bt.model == model) & (bt.h == h) & (bt.target_hour == target_hour)]["resid"]
    if len(g) < 30:
        g = bt[(bt.model == model) & (bt.h == h)]["resid"]
    return float(((pred + g.to_numpy()) >= threshold).mean())
