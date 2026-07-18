# gb-battery (backend)

Python engine for the GB Battery Co-Optimisation Terminal: DST-aware settlement
calendar, Pyomo/HiGHS battery optimiser, Elexon/NESO data adapters with lineage,
point-in-time replay engine, forecasting and backtesting, exposed via FastAPI.

Research & decision-support only — no live trading, no asset control. See the
repository root [README](../README.md) and [docs/](../docs/) for full documentation.

```bash
pip install -e ".[dev]"
pytest
uvicorn gb_battery.api.main:app --port 8000
```
