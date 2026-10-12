"""FastAPI service: weather risk -> crew recommendation.

    uvicorn app.main:app --reload        (from the repo root, with PYTHONPATH=src)
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wwp import config as C  # noqa: E402
from wwp import data_weather as W  # noqa: E402
from wwp import features as F  # noqa: E402
from wwp.decision import recommend  # noqa: E402
from wwp.workforce import is_harvest_day, new_ready_bins  # noqa: E402

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Weather-Aware Workforce Planning",
              description="Decision support: tomorrow's rain risk -> picking crew size.")
_bundle: dict | None = None


def bundle() -> dict:
    global _bundle
    if _bundle is None:
        path = C.MODEL_DIR / "bundle.joblib"
        if not path.exists():
            raise HTTPException(503, "Model not trained. Run: python -m wwp.pipeline")
        _bundle = joblib.load(path)
    return _bundle


class FarmOverrides(BaseModel):
    crew_pool: int | None = Field(None, ge=1, le=500)
    packhouse_slots: int | None = Field(None, ge=0, le=500)
    wage: float | None = Field(None, gt=0)
    min_paid_hours: float | None = Field(None, ge=0, le=12)
    delay_cost: float | None = Field(None, ge=0)


class WhatIf(FarmOverrides):
    p_wet1: float = Field(..., ge=0, le=1, description="P(>= 1 wet working hour)")
    p_washout: float = Field(..., ge=0, le=1, description="P(>= 4 wet working hours)")
    p_wet7: float = Field(0.0, ge=0, le=1, description="P(>= 7 wet working hours)")
    ready_bins: float = Field(..., ge=0)
    date: str | None = None


def _farm(o: FarmOverrides):
    return replace(C.FARM, **{k: v for k, v in o.model_dump().items()
                              if k in C.FARM.__dataclass_fields__ and v is not None})


def _ready_and_rate(d: pd.Timestamp, ready: float | None, farm):
    impact = bundle()["impact"]
    in_season = is_harvest_day(d)
    if ready is None:
        ready = new_ready_bins(d, farm) if in_season else 0.6 * farm.peak_ready_bins
    rate = impact.rate(d) if in_season else impact.rate_main
    return float(ready), rate, in_season


def _plan(d: pd.Timestamp, risk: dict, ready: float | None, o: FarmOverrides) -> dict:
    farm = _farm(o)
    ready, rate, in_season = _ready_and_rate(d, ready, farm)
    rec = recommend(risk, ready, rate, farm, bundle()["impact"])
    note = None if in_season else ("Outside the apple harvest (1 Feb - 15 May): "
                                   "ready bins is an example value; set your own.")
    return {"date": str(d.date()), "risk": risk, "ready_bins": round(ready, 1),
            "season_note": note, **rec}


def _risk(row: pd.Series) -> dict:
    return {"p_rain": round(float(row.p_rain), 3), "p_wet1": round(float(row.p_wet1), 3),
            "p_washout": round(float(row.p_washout), 3), "p_wet7": round(float(row.p_wet7), 3),
            "expected_mm": round(float(row.expected_mm), 1),
            "forecast_mm": None if pd.isna(row.nwp_rain_mm) else round(float(row.nwp_rain_mm), 1),
            "forecast_wet_hours": None if pd.isna(row.get("nwp_rain_hours", np.nan))
            else int(row.nwp_rain_hours),
            "model": row.source}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (C.MODEL_DIR / "bundle.joblib").exists()}


@app.get("/api/dates")
def dates():
    replay = bundle()["replay"]
    harvest = [d for d in replay.index if is_harvest_day(d) and d.dayofweek != 6]
    return {"site": C.SITE_NAME, "dates": [str(d.date()) for d in harvest]}


@app.get("/api/plan")
def plan_replay(date: str, ready_bins: float | None = Query(None, ge=0),
                crew_pool: int | None = Query(None, ge=1, le=500),
                min_paid_hours: float | None = Query(None, ge=0, le=12),
                delay_cost: float | None = Query(None, ge=0)):
    """Replay a past day: the model's forecast as it stood the evening before, plus
    what actually happened (so you can judge the call)."""
    replay = bundle()["replay"]
    d = pd.Timestamp(date)
    if d not in replay.index:
        raise HTTPException(404, f"No replay data for {date}")
    row = replay.loc[d]
    o = FarmOverrides(crew_pool=crew_pool, min_paid_hours=min_paid_hours, delay_cost=delay_cost)
    out = _plan(d, _risk(row), ready_bins, o)
    out["actual"] = {"rain_mm": round(float(row.obs_rain_mm), 1),
                     "wet_hours": int(row.obs_rain_hours),
                     "washout": bool(row.y_washout)}
    return out


@app.get("/api/plan/live")
def plan_live(ready_bins: float | None = Query(None, ge=0),
              crew_pool: int | None = Query(None, ge=1, le=500),
              min_paid_hours: float | None = Query(None, ge=0, le=12),
              delay_cost: float | None = Query(None, ge=0)):
    """Tomorrow, from the live Open-Meteo forecast."""
    try:
        obs, nwp = W.fetch_live()
    except Exception as exc:  # network down -> say so, the replay mode still works
        raise HTTPException(502, f"Live forecast unavailable: {exc}") from exc
    hist = F.history_features(obs)
    fc = F.nwp_features(nwp)
    tomorrow = pd.Timestamp.now(tz=C.TIMEZONE).normalize().tz_localize(None) + pd.Timedelta(days=1)
    if tomorrow not in hist.index or tomorrow not in fc.index:
        raise HTTPException(502, "Live forecast did not cover tomorrow")
    X = hist.loc[[tomorrow]].join(fc.loc[[tomorrow]])
    row = bundle()["serving"].predict(X).iloc[0]
    o = FarmOverrides(crew_pool=crew_pool, min_paid_hours=min_paid_hours, delay_cost=delay_cost)
    return _plan(tomorrow, _risk(row), ready_bins, o)


@app.post("/api/recommend")
def what_if(body: WhatIf):
    """What-if: supply your own probabilities and farm numbers."""
    d = pd.Timestamp(body.date) if body.date else pd.Timestamp("2026-03-15")
    risk = {"p_rain": None, "p_wet1": body.p_wet1, "p_washout": body.p_washout,
            "p_wet7": body.p_wet7, "expected_mm": None, "forecast_mm": None,
            "forecast_wet_hours": None, "model": "what-if (user supplied)"}
    return _plan(d, risk, body.ready_bins, body)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
