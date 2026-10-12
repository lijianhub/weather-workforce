"""Turn hourly weather into one row per *target day* T.

The roster for day T is decided at 18:00 on day T-1, so:
  * history features use observations up to 17:59 on T-1 (never later),
  * NWP features use the day-ahead forecast for T (issued before T-1 evening),
  * targets use observations during working hours on T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _work_hours(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    return df[(h >= C.WORK_START_HOUR) & (h < C.WORK_END_HOUR)]


def targets(obs: pd.DataFrame) -> pd.DataFrame:
    """What actually happened during working hours on each day."""
    w = _work_hours(obs)
    g = w.groupby(w.index.normalize())
    out = pd.DataFrame({
        "obs_rain_mm": g["precipitation"].sum(),
        "obs_rain_hours": g["precipitation"].apply(lambda s: (s >= C.WET_HOUR_MM).sum()),
        "obs_gust_max": g["wind_gusts_10m"].max(),
        "obs_temp_max": g["temperature_2m"].max(),
    })
    out["y_rain"] = (out["obs_rain_mm"] >= C.RAIN_MM).astype(int)
    out["y_wet1"] = (out["obs_rain_hours"] >= 1).astype(int)
    out["y_washout"] = (out["obs_rain_hours"] >= C.WASHOUT_HOURS).astype(int)
    out["y_wet7"] = (out["obs_rain_hours"] >= 7).astype(int)
    out.index.name = "date"
    return out


def history_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Features known at the 18:00 decision time on T-1, indexed by T."""
    obs = obs.copy()
    rad = np.deg2rad(obs["wind_direction_10m"])
    obs["wind_u"] = -obs["wind_speed_10m"] * np.sin(rad)   # +ve = westerly flow
    obs["wind_v"] = -obs["wind_speed_10m"] * np.cos(rad)   # +ve = southerly flow
    day = obs.index.normalize()

    before_cut = obs[obs.index.hour < C.INFO_CUTOFF_HOUR]
    g = before_cut.groupby(before_cut.index.normalize())
    at = lambda hour: obs[obs.index.hour == hour].set_index(day[obs.index.hour == hour])
    h17, h05 = at(C.INFO_CUTOFF_HOUR - 1), at(5)

    d = pd.DataFrame({
        "today_rain_mm": g["precipitation"].sum(),
        "today_rain_hours": g["precipitation"].apply(lambda s: (s >= 0.3).sum()),
        "today_temp_mean": g["temperature_2m"].mean(),
        "today_temp_max": g["temperature_2m"].max(),
        "today_rh_mean": g["relative_humidity_2m"].mean(),
        "today_cloud_mean": g["cloud_cover"].mean(),
        "today_gust_max": g["wind_gusts_10m"].max(),
        "rh_17": h17["relative_humidity_2m"],
        "cloud_17": h17["cloud_cover"],
        "pressure_17": h17["pressure_msl"],
        "pressure_tend_12h": h17["pressure_msl"] - h05["pressure_msl"],
        "wind_u_17": h17["wind_u"],
        "wind_v_17": h17["wind_v"],
    })
    d["pressure_tend_24h"] = d["pressure_17"].diff(1)

    full_day_rain = obs["precipitation"].groupby(day).sum()
    d["rain_yesterday_mm"] = full_day_rain.shift(1)
    d["rain_prev_7d_mm"] = full_day_rain.shift(1).rolling(7).sum()

    # Re-index from decision day (T-1) to target day T.
    d.index = d.index + pd.Timedelta(days=1)
    doy = d.index.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d.index.name = "date"
    return d


def nwp_features(nwp: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead forecast for working hours on T, indexed by T."""
    w = _work_hours(nwp)
    g = w.groupby(w.index.normalize())
    out = pd.DataFrame({
        "nwp_rain_mm": g["precipitation"].sum(),
        "nwp_rain_max_hourly": g["precipitation"].max(),
        "nwp_rain_hours": g["precipitation"].apply(lambda s: (s >= C.WET_HOUR_MM).sum()),
        "nwp_gust_max": g["wind_gusts_10m"].max(),
        "nwp_temp_max": g["temperature_2m"].max(),
        "nwp_cloud_mean": g["cloud_cover"].mean(),
        "nwp_rh_mean": g["relative_humidity_2m"].mean(),
    })
    counts = g["precipitation"].count()
    out = out[counts == C.WORK_END_HOUR - C.WORK_START_HOUR]   # drop partial days
    out.index.name = "date"
    return out


HISTORY_COLS = [
    "today_rain_mm", "today_rain_hours", "today_temp_mean", "today_temp_max",
    "today_rh_mean", "today_cloud_mean", "today_gust_max", "rh_17", "cloud_17",
    "pressure_17", "pressure_tend_12h", "pressure_tend_24h", "wind_u_17", "wind_v_17",
    "rain_yesterday_mm", "rain_prev_7d_mm", "doy_sin", "doy_cos",
]
NWP_COLS = ["nwp_rain_mm", "nwp_rain_max_hourly", "nwp_rain_hours", "nwp_gust_max",
            "nwp_temp_max", "nwp_cloud_mean", "nwp_rh_mean"]
# The forecast-correction (MOS) model: forecast + a few local signals.
MOS_COLS = NWP_COLS + ["today_rain_mm", "rh_17", "pressure_tend_12h",
                       "wind_u_17", "wind_v_17", "doy_sin", "doy_cos"]


def build_dataset(obs: pd.DataFrame, nwp: pd.DataFrame | None) -> pd.DataFrame:
    df = history_features(obs).join(targets(obs), how="inner")
    if nwp is not None:
        df = df.join(nwp_features(nwp), how="left")
    return df.dropna(subset=HISTORY_COLS)
