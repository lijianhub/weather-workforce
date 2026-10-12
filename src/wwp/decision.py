"""Turn a weather risk forecast into a crew recommendation.

The weather model gives a distribution over how many working hours will be wet:

    P(0 wet hours), P(1-3), P(4-6), P(7+)

The impact model knows how many productive hours crews really got in each band. For
every crew size n = 0..pool we compute the EXPECTED avoidable cost

    E[cost(n)] = sum over bands  P(band) * mean cost(n | productive hours seen in band)

and pick the cheapest. It uses the probabilities directly (no 50% threshold), so the
recommendation is only as good as the model's calibration.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .config import FarmParams
from .workforce import ImpactModel, day_outcome


def band_probs(p_wet1: float, p_washout: float, p_wet7: float) -> dict[str, float]:
    p_washout = min(p_washout, p_wet1)
    p_wet7 = min(p_wet7, p_washout)
    return {"0": 1 - p_wet1, "1-3": p_wet1 - p_washout, "4-6": p_washout - p_wet7, "7+": p_wet7}


def expected_outcomes(probs: dict[str, float], ready: float, rate: float,
                      farm: FarmParams, impact: ImpactModel) -> dict[str, np.ndarray]:
    n = np.arange(farm.crew_pool + 1)[:, None]
    total: dict[str, np.ndarray] = {}
    for band, p in probs.items():
        hours = impact.hours[band][None, :]
        for k, v in day_outcome(n, hours, rate, ready, farm).items():
            v = np.broadcast_to(v, (n.shape[0], hours.size)).mean(axis=1)
            total[k] = total.get(k, 0) + p * v
    return total


DRY = {"0": 1.0, "1-3": 0.0, "4-6": 0.0, "7+": 0.0}


def best_crew(probs, ready, rate, farm, impact) -> int:
    return int(np.argmin(expected_outcomes(probs, ready, rate, farm, impact)["total_cost"]))


def recommend(risk: dict, ready: float, rate: float, farm: FarmParams,
              impact: ImpactModel) -> dict:
    probs = band_probs(risk["p_wet1"], risk["p_washout"], risk["p_wet7"])
    exp = expected_outcomes(probs, ready, rate, farm, impact)
    best = int(np.argmin(exp["total_cost"]))
    dry_crew = best_crew(DRY, ready, rate, farm, impact)
    pool = farm.crew_pool

    def option(name, n):
        return {"option": name, "outdoor_crew": n,
                "packhouse_crew": int(min(pool - n, farm.packhouse_slots)),
                "not_rostered": int(max(pool - n - farm.packhouse_slots, 0)),
                "expected_bins_picked": round(float(exp["picked"][n]), 1),
                "expected_bins_left": round(float(exp["unpicked"][n]), 1),
                "expected_idle_cost": round(float(exp["idle_cost"][n])),
                "overhead_cost": round(float(exp["overhead_cost"][n])),
                "expected_delay_cost": round(float(exp["delay_cost"][n])),
                "expected_total_cost": round(float(exp["total_cost"][n])),
                "worker_hours_offered": round(float(exp["worker_hours_offered"][n]))}

    options = [option("Full crew (status quo)", pool),
               option("Crew for a dry day (ignore weather)", dry_crew),
               option("AI recommendation", best),
               option("Reschedule outdoor work", 0)]
    saving = options[1]["expected_total_cost"] - options[2]["expected_total_cost"]

    if best <= 0.25 * dry_crew:
        headline = "Likely washout: stand the picking crew down and use the packhouse."
    elif best < 0.9 * dry_crew:
        headline = "Rain will cost working hours: run a reduced picking crew."
    elif probs["4-6"] + probs["7+"] >= 0.3:
        headline = ("Real risk of losing half the day, but with this much ripe fruit calling "
                    "the crew is still the better bet. Re-check the 6 am forecast.")
    elif best > 1.1 * dry_crew:
        headline = "Showers may cost an hour or two: add a buffer to the crew."
    else:
        headline = "Good picking weather: roster the crew the ripe fruit needs."
    reasons = [
        f"Chance of wet working hours tomorrow: none {probs['0']:.0%}, 1-3 h "
        f"{probs['1-3']:.0%}, 4-6 h {probs['4-6']:.0%}, 7+ h {probs['7+']:.0%}.",
        f"{ready:.0f} bins are ready; a picker averages {rate:.2f} bins per productive hour, "
        f"so a dry day needs {dry_crew} pickers.",
        f"Called pickers are paid at least {farm.min_paid_hours:g} h "
        f"(${farm.wage * farm.min_paid_hours:,.0f}) plus ${farm.per_worker_cost:g} overhead, "
        "even if rain stops picking.",
        f"Each ready bin left on the tree costs about ${farm.delay_cost:g} per day.",
        f"Expected saving versus ignoring the weather: ${saving:,.0f}.",
    ]
    if options[2]["packhouse_crew"]:
        reasons.append(f"{options[2]['packhouse_crew']} workers can move to packhouse work "
                       "so they still get hours.")
    return {"recommended_crew": best, "dry_day_crew": dry_crew, "headline": headline,
            "reasons": reasons, "wet_hour_bands": {k: round(v, 3) for k, v in probs.items()},
            "options": options, "cost_curve": [round(float(c)) for c in exp["total_cost"]],
            "farm": asdict(farm)}
