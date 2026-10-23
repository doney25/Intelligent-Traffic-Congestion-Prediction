"""Data loading and feature preparation (mirrors the notebook's cleaning steps)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "Metro_Interstate_Traffic_Volume_snow_updated.csv"
LOOKBACK = 24
FEATURES = ["hour", "dow", "month", "is_weekend", "is_holiday", "temp_c", "rain", "snow", "clouds",
            "now", "lag1", "lag2", "lag24", "roll3", "roll6"]


@dataclass
class TrafficData:
    full: pd.Series              # hourly volume, NaN where the sensor has gaps
    weather: pd.DataFrame        # hourly weather aligned to `full`
    split_ts: pd.Timestamp       # first timestamp of the notebook's 20% test period
    features: pd.DataFrame       # model features per hour (NaN where not computable)
    valid_asof: pd.DatetimeIndex # hours with 25 contiguous hours of history

    # -- helpers ---------------------------------------------------------
    @property
    def first(self):
        return self.full.index[0]

    @property
    def last(self):
        return self.full.dropna().index[-1]

    def windows(self, times) -> np.ndarray:
        """(n, 24) array of the 24 hourly volumes ending at each time (inclusive)."""
        times = pd.DatetimeIndex(times)
        offs = pd.to_timedelta(np.arange(-LOOKBACK + 1, 1), unit="h")
        idx = (times.values[:, None] + offs.values[None, :]).ravel()
        vals = self.full.reindex(pd.DatetimeIndex(idx)).to_numpy().reshape(len(times), LOOKBACK)
        return vals

    def value_at(self, ts):
        return self.full.get(pd.Timestamp(ts), np.nan)

    def feature_matrix(self, times) -> np.ndarray:
        return self.features.reindex(pd.DatetimeIndex(times))[FEATURES].to_numpy()


def load_data() -> TrafficData:
    df = pd.read_csv(CSV_PATH).drop_duplicates()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["holiday_flag"] = df["holiday"].notna().astype(int)

    series = (df.groupby("date_time")["traffic_volume"].mean().sort_index().resample("h").mean())
    series = series.ffill(limit=3).dropna()

    wx = df.groupby("date_time").agg(temp_c=("temp", lambda x: x.mean() - 273.15), rain=("rain_1h", "mean"),
                                      snow=("snow_1h", "mean"), clouds=("clouds_all", "mean"),
                                      is_holiday=("holiday_flag", "max"))
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="h")
    full = series.reindex(full_idx)
    weather = wx.reindex(full_idx).ffill(limit=3)

    split_ts = series.index[int(len(series) * 0.8)]

    f = pd.DataFrame(index=full_idx)
    f["hour"] = full_idx.hour
    f["dow"] = full_idx.dayofweek
    f["month"] = full_idx.month
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["is_holiday"] = weather["is_holiday"]
    f["temp_c"], f["rain"], f["snow"], f["clouds"] = (weather[c] for c in ("temp_c", "rain", "snow", "clouds"))
    f["now"] = full
    f["lag1"], f["lag2"], f["lag24"] = full.shift(1), full.shift(2), full.shift(24)
    f["roll3"], f["roll6"] = full.rolling(3).mean(), full.rolling(6).mean()

    ok = full.notna()
    contiguous = ok.rolling(LOOKBACK + 1).sum() == (LOOKBACK + 1)
    valid = full_idx[contiguous.to_numpy()]
    return TrafficData(full, weather, split_ts, f, valid)


def hourly_profile(full: pd.Series, day_type: str) -> pd.Series:
    """Average volume by hour for 'Weekday' or 'Weekend'."""
    s = full.dropna()
    mask = s.index.dayofweek >= 5 if day_type == "Weekend" else s.index.dayofweek < 5
    return s[mask].groupby(s[mask].index.hour).mean().reindex(range(24))
