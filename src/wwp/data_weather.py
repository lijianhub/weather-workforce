"""Download and cache hourly weather from Open-Meteo (free, no API key).

Two sources:
  * observed  - ERA5 reanalysis from the archive API (our "what actually happened")
  * nwp       - the day-ahead forecast as it was issued the day before
                (Open-Meteo previous-runs API, `*_previous_day1` variables)
"""
from __future__ import annotations

import pandas as pd
import requests

from . import config as C

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OBS_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]
NWP_VARS = ["precipitation", "temperature_2m", "wind_gusts_10m", "cloud_cover",
            "relative_humidity_2m"]


def _get_hourly(url: str, params: dict) -> pd.DataFrame:
    base = {"latitude": C.LATITUDE, "longitude": C.LONGITUDE, "timezone": C.TIMEZONE}
    resp = requests.get(url, params={**base, **params}, timeout=120)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _year_chunks(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for year in range(s.year, e.year + 1):
        yield max(s, pd.Timestamp(f"{year}-01-01")), min(e, pd.Timestamp(f"{year}-12-31"))


def fetch_observed(refresh: bool = False) -> pd.DataFrame:
    path = C.RAW_DIR / "observed_hourly.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="time", parse_dates=True)
    parts = []
    for s, e in _year_chunks(C.HISTORY_START, C.HISTORY_END):
        print(f"  observed {s.date()} -> {e.date()}")
        parts.append(_get_hourly(ARCHIVE_URL, {
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "hourly": ",".join(OBS_VARS)}))
    df = pd.concat(parts).sort_index()
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def fetch_nwp_day_ahead(refresh: bool = False) -> pd.DataFrame:
    path = C.RAW_DIR / "nwp_day_ahead_hourly.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="time", parse_dates=True)
    parts = []
    names = [f"{v}_previous_day1" for v in NWP_VARS]
    for s, e in _year_chunks(C.NWP_START, C.HISTORY_END):
        print(f"  nwp day-ahead {s.date()} -> {e.date()}")
        parts.append(_get_hourly(PREVIOUS_RUNS_URL, {
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "hourly": ",".join(names)}))
    df = pd.concat(parts).sort_index()
    df.columns = [c.replace("_previous_day1", "") for c in df.columns]
    df = df.dropna(how="all")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def fetch_live() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recent observed-like hours plus tomorrow's forecast, for the live endpoint.

    Note: live "history" comes from the forecast model's own recent analysis, not
    ERA5. That is a real training/serving skew, documented in the README.
    """
    df = _get_hourly(FORECAST_URL, {
        "hourly": ",".join(sorted(set(OBS_VARS) | set(NWP_VARS))),
        "past_days": 7, "forecast_days": 3})
    return df[OBS_VARS], df[NWP_VARS]


if __name__ == "__main__":
    obs = fetch_observed()
    nwp = fetch_nwp_day_ahead()
    print(obs.shape, obs.index.min(), obs.index.max())
    print(nwp.shape, nwp.index.min(), nwp.index.max())
