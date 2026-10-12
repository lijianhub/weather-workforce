"""Synthetic orchard workforce data and the weather -> work impact model.

There is no public picking dataset, so we SIMULATE one, driven by the *real* hourly
weather at the orchard. The rules (rain stops picking, gusts stop ladder work, heat
slows pickers) are stated openly below. The pipeline is the point: with real
timesheet + harvest-tracking exports, `generate_harvest_log` is replaced by a loader
and everything downstream stays the same.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .config import FarmParams

RAIN_STOP_MM = C.WET_HOUR_MM  # a wet hour stops picking ...
DRYING_HOURS = 1         # ... and the next hour too (wet fruit, slippery ladders)
GUST_STOP_KMH = 65       # ladder work stops in strong gusts
HEAT_C = 30              # above this, pickers slow down
HEAT_FACTOR = 0.7


# ---------- season & simulation rules -------------------------------------------------

def is_harvest_day(d: pd.Timestamp) -> bool:
    start = pd.Timestamp(d.year, *C.HARVEST_START)
    end = pd.Timestamp(d.year, *C.HARVEST_END)
    return start <= d <= end


def season_fraction(d: pd.Timestamp) -> float:
    start = pd.Timestamp(d.year, *C.HARVEST_START)
    end = pd.Timestamp(d.year, *C.HARVEST_END)
    return (d - start).days / (end - start).days


def new_ready_bins(d: pd.Timestamp, farm: FarmParams) -> float:
    """Bins that ripen on day d: ramps up, peaks mid-season, tails off."""
    x = season_fraction(d)
    return farm.peak_ready_bins * (0.25 + 0.75 * np.sin(np.pi * x) ** 1.5)


def season_rate_factor(d: pd.Timestamp) -> float:
    """New crews are slower for the first two weeks."""
    return 0.85 if season_fraction(d) < 0.14 else 1.0


def productive_hours(hours: pd.DataFrame, farm: FarmParams, rng: np.random.Generator) -> float:
    """Productive picking hours available in the 07:00-17:00 window on one day."""
    w = hours[(hours.index.hour >= C.WORK_START_HOUR) & (hours.index.hour < C.WORK_END_HOUR)]
    wet = (w["precipitation"] >= RAIN_STOP_MM).to_numpy()
    stopped = wet.copy()
    for k in range(1, DRYING_HOURS + 1):
        stopped[k:] |= wet[:-k]
    stopped |= (w["wind_gusts_10m"] >= GUST_STOP_KMH).to_numpy()
    weight = np.where(w["temperature_2m"].to_numpy() >= HEAT_C, HEAT_FACTOR, 1.0)
    available = float((weight * ~stopped).sum())
    available *= rng.normal(1.0, 0.05)              # unmodelled site variation
    return float(np.clip(min(available, farm.shift_hours), 0, farm.shift_hours))


def day_outcome(n: np.ndarray | int, H: np.ndarray | float, rate: np.ndarray | float,
                ready: float, farm: FarmParams) -> dict:
    """Outcome of calling n pickers when the day turns out to have H productive hours.
    `total_cost` is the avoidable cost: idle paid time + overhead on unproductive time
    + fruit delay.

    Vectorised: n can be a column vector and H/rate row vectors (scenarios).
    """
    n = np.asarray(n, dtype=float)
    capacity = n * H * rate
    picked = np.minimum(capacity, ready)
    with np.errstate(divide="ignore", invalid="ignore"):
        hours_needed = np.where(n > 0, ready / (n * rate), 0.0)
    worked = np.minimum(H, hours_needed)
    paid_hours = n * np.maximum(farm.min_paid_hours, worked)
    paid_hours = np.where(n > 0, paid_hours, 0.0)
    labour = farm.wage * paid_hours
    # Picking labour is spent eventually whatever we decide, so the cost a roster
    # decision can avoid is paid time with no picking (guaranteed hours on a washout
    # or with too little ripe fruit) plus the cost of leaving ripe fruit a day longer.
    idle = farm.wage * (paid_hours - n * worked)
    # Same logic for the daily per-worker overhead: a full productive day is needed
    # eventually anyway; only the unproductive share of each called picker is waste.
    overhead = farm.per_worker_cost * n * (1 - worked / farm.shift_hours)
    unpicked = ready - picked
    delay = farm.delay_cost * unpicked
    packhouse = np.minimum(farm.crew_pool - n, farm.packhouse_slots)
    return {"picked": picked, "unpicked": unpicked, "paid_hours": paid_hours,
            "labour_cost": labour, "idle_cost": idle, "overhead_cost": overhead,
            "delay_cost": delay, "total_cost": idle + overhead + delay,
            "packhouse_workers": packhouse,
            "worker_hours_offered": paid_hours + packhouse * farm.shift_hours}


# ---------- historical workforce-platform-style log -----------------------------------

def generate_harvest_log(obs: pd.DataFrame, daily: pd.DataFrame, years: range,
                         farm: FarmParams = C.FARM, seed: int = 7) -> pd.DataFrame:
    """One row per picking day, like a timesheet + harvest-tracking export.

    The historical manager mostly calls the full crew, trimming it a bit when it was
    raining at roster time. That gives the log realistic variety.
    """
    rng = np.random.default_rng(seed)
    by_day = {d: g for d, g in obs.groupby(obs.index.normalize())}
    rows, backlog = [], 0.0
    for d in pd.date_range(f"{years.start}-01-01", f"{years.stop - 1}-12-31"):
        if not is_harvest_day(d):
            backlog = 0.0
            continue
        ready = backlog + new_ready_bins(d, farm)
        if d.dayofweek == 6:                         # Sunday: no picking
            backlog = ready
            continue
        H = productive_hours(by_day[d], farm, rng)
        rate = farm.pick_rate * season_rate_factor(d) * rng.lognormal(0, 0.08)
        wet_evening = daily.loc[d, "today_rain_mm"] >= C.RAIN_MM if d in daily.index else False
        n = int(round(farm.crew_pool * rng.uniform(0.85, 1.0) * (0.7 if wet_evening else 1.0)))
        out = day_outcome(n, H, rate, ready, farm)
        backlog = float(out["unpicked"])
        rows.append({
            "date": d, "farm": "HB-Orchard-01", "crop": "apples", "activity": "harvest",
            "crew_called": n, "productive_hours": round(H, 2),
            "picker_hours_paid": round(float(out["paid_hours"]), 1),
            "bins_ready": round(ready, 1), "bins_picked": round(float(out["picked"]), 1),
            "bins_per_productive_hour": round(rate, 3),
            "work_hours_rain_mm": round(float(daily.loc[d, "obs_rain_mm"]), 2)
            if d in daily.index else np.nan,
            "work_hours_wet_hours": int(daily.loc[d, "obs_rain_hours"])
            if d in daily.index else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


# ---------- learned impact model -----------------------------------------------------

BANDS = ("0", "1-3", "4-6", "7+")


def wet_band(wet_hours) -> np.ndarray:
    """How many of the 10 working hours were wet, in four bands."""
    h = np.asarray(wet_hours)
    return np.select([h >= 7, h >= C.WASHOUT_HOURS, h >= 1], ["7+", "4-6", "1-3"], "0")


class ImpactModel:
    """Learned from the harvest log: how many productive hours a crew actually gets
    for each band of wet working hours, and the picking rate by season stage."""

    def fit(self, log: pd.DataFrame) -> "ImpactModel":
        log = log.dropna(subset=["work_hours_wet_hours"])
        band = wet_band(log["work_hours_wet_hours"])
        self.hours = {b: log.loc[band == b, "productive_hours"].to_numpy() for b in BANDS}
        early = log.index.map(lambda d: season_fraction(d) < 0.14)
        self.rate_early = float(log.loc[early, "bins_per_productive_hour"].median())
        self.rate_main = float(log.loc[~early, "bins_per_productive_hour"].median())
        return self

    def rate(self, d: pd.Timestamp) -> float:
        return self.rate_early if season_fraction(d) < 0.14 else self.rate_main

    def summary(self) -> dict:
        return {c: {"days": int(len(h)), "mean_hours": round(float(h.mean()), 2),
                    "p10_hours": round(float(np.percentile(h, 10)), 2),
                    "share_washout_lt2h": round(float((h < 2).mean()), 3)}
                for c, h in self.hours.items()}
