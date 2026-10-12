"""Project-wide settings. Every business number here is an assumption, kept in one
place so it can be swapped for a real customer's figures."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Orchard location: Hastings, Hawke's Bay (NZ's main apple region).
SITE_NAME = "Hastings, Hawke's Bay"
LATITUDE = -39.64
LONGITUDE = 176.84
TIMEZONE = "Pacific/Auckland"

# Observed weather (ERA5 reanalysis) history.
HISTORY_START = "2010-01-01"
HISTORY_END = "2026-09-30"
# Archived day-ahead NWP forecasts exist from 2024 for this location.
NWP_START = "2024-01-01"

# Time-based split. No shuffling: the future must never leak into training.
TRAIN_END = "2023-12-31"   # weather models (history features) train on 2010-2023
VALID_END = "2024-12-31"   # 2024: early stopping, calibration, MOS training
# test = 2025-01-01 onwards (includes the 2025 and 2026 apple harvests)

# Picking happens 07:00-17:00. Rain outside these hours barely matters to a crew.
WORK_START_HOUR = 7
WORK_END_HOUR = 17          # exclusive -> 10 working hours
# The roster for tomorrow is set at 18:00 today, so today's features stop at 17:59.
INFO_CUTOFF_HOUR = 18

RAIN_MM = 1.0               # "rain day": >= 1 mm during work hours
WET_HOUR_MM = 0.3           # an hour with >= 0.3 mm is a wet hour (picking stops)
WASHOUT_HOURS = 4           # "washout": >= 4 wet work hours -> most of the day lost.
# Crews lose HOURS, not millimetres: a single 10 mm burst at 4pm is not a washout.

# Harvest season (apples): 1 Feb - 15 May, picking Monday to Saturday.
HARVEST_START = (2, 1)
HARVEST_END = (5, 15)


@dataclass(frozen=True)
class FarmParams:
    """Economics of one orchard crew. Assumed values, not real customer data."""
    crew_pool: int = 40              # pickers available on the roster
    packhouse_slots: int = 12        # indoor work available if outdoor work is cut
    shift_hours: float = 8.0
    wage: float = 30.0               # loaded hourly cost (wage + on-costs), NZD
    min_paid_hours: float = 4.0      # guaranteed hours once a worker is called in
    per_worker_cost: float = 25.0    # daily overhead per picker called: transport, supervision, admin
    pick_rate: float = 0.8           # bins per picker per productive hour
    delay_cost: float = 15.0         # value lost per ripe bin left on the tree a day (~2-3% of bin value)
    peak_ready_bins: float = 180.0   # bins ripening per day at peak (~70% of full-crew capacity)


FARM = FarmParams()
