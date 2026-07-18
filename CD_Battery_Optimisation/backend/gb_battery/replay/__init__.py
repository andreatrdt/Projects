"""Rolling-horizon replay: point-in-time data, forecasting and decision simulation.

This package answers the question the rest of the app must never fudge:

    "At every historical or live decision timestamp, what information was
    available, what did the model forecast, what action did it choose, and how
    did that decision perform once the actual outcome became known?"

Design rules enforced here:

* Every record carries a publication timestamp; the :class:`~gb_battery.replay.pit.PITDataStore`
  is the *only* gateway to data during a replay and always filters
  ``published_at <= as_of``.
* Forecasts are produced out-of-sample at each decision time; provenance
  (observed / published_forecast / model_forecast / reconstructed / synthetic /
  assumed) is retained end-to-end.
* Only the first Settlement Period of each proposed schedule is executed; SoC
  and cumulative P&L persist across steps.
* Perfect foresight exists only as an explicitly-labelled benchmark.
"""

from gb_battery.replay.engine import ReplayEngine, ReplayOptions
from gb_battery.replay.pit import PITDataStore, PITViolation
from gb_battery.replay.records import ForecastRecord, ObservationRecord, Provenance

__all__ = [
    "ForecastRecord",
    "ObservationRecord",
    "PITDataStore",
    "PITViolation",
    "Provenance",
    "ReplayEngine",
    "ReplayOptions",
]
