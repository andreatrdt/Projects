"""Command-line entry points for the GB Battery Co-Optimisation Terminal.

Usage:
    python -m gb_battery.cli freeze-sample
    python -m gb_battery.cli ingest-demo [--day YYYY-MM-DD] [--offline]
    python -m gb_battery.cli optimise-demo [--day YYYY-MM-DD]
    python -m gb_battery.cli backtest [--days N]
"""

from __future__ import annotations

import argparse
from datetime import date

from gb_battery.battery.config import BatteryConfig


def _cmd_freeze_sample(_args: argparse.Namespace) -> int:
    from gb_battery.demo.sample_data import freeze_sample

    path = freeze_sample()
    print(f"Frozen synthetic sample written to {path}")
    return 0


def _cmd_ingest_demo(args: argparse.Namespace) -> int:
    from gb_battery.data.market_snapshot import build_market_snapshot
    from gb_battery.data.settings import DataSettings

    day = date.fromisoformat(args.day) if args.day else date(2025, 1, 14)
    settings = DataSettings(offline=args.offline)
    snap = build_market_snapshot(day, settings=settings)
    print(f"Market snapshot for {day} ({len(snap.frame)} settlement periods)")
    for s in snap.statuses:
        flag = "OK " if s.ok else "!! "
        print(f"  {flag}{s.source:14s} kind={s.kind.value:9s} {s.detail}")
    for w in snap.warnings:
        print(f"  warning: {w}")
    return 0


def _cmd_optimise_demo(args: argparse.Namespace) -> int:
    from gb_battery.demo.scenarios import synthetic_day_inputs
    from gb_battery.optimiser.deterministic import optimise

    day = date.fromisoformat(args.day) if args.day else date(2025, 1, 15)
    res = optimise(BatteryConfig(), synthetic_day_inputs(day), compute_marginals=False)
    print(f"Optimised {day}: status={res.status} solver={res.solver}")
    print(f"  Expected P&L: GBP {res.total_expected_pnl_gbp:,.0f}  |  cycles: {res.full_cycle_equivalents}")
    print(f"  wholesale={res.total_wholesale_pnl_gbp:,.0f} service={res.total_service_pnl_gbp:,.0f} "
          f"bm={res.total_bm_activation_pnl_gbp:,.0f} degradation={res.total_degradation_cost_gbp:,.0f}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from gb_battery.backtest import compare_strategies
    from gb_battery.demo.sample_data import generate_synthetic_history

    hist = generate_synthetic_history(days=args.days)
    cfg = BatteryConfig(degradation_cost_gbp_per_mwh_throughput=2.0, minimum_terminal_soc_mwh=10.0)
    cmp = compare_strategies(cfg, hist)
    print(f"Backtest over {args.days} days (perfect-foresight upper bound: "
          f"GBP {cmp['perfect_foresight_pnl_gbp']:,.0f})")
    print(f"  {'strategy':24s} {'PnL(GBP)':>10s} {'cycles':>7s} {'MAE':>6s} {'capture%':>8s}")
    for row in cmp["table"]:
        print(f"  {row['strategy']:24s} {row['total_pnl_gbp']:10,.0f} "
              f"{row['full_cycle_equivalents']:7.2f} {row.get('price_forecast_mae', 0):6.1f} "
              f"{row.get('capture_of_perfect_pct', '-'):>8}")
    for a in cmp["leakage_audit"]:
        print(f"  leakage-audit [{a['check']}]: {'PASS' if a['ok'] else 'FAIL'} — {a['detail']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gb-battery", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("freeze-sample", help="Regenerate the frozen synthetic sample").set_defaults(func=_cmd_freeze_sample)

    p_ing = sub.add_parser("ingest-demo", help="Build a market snapshot (offline-capable)")
    p_ing.add_argument("--day")
    p_ing.add_argument("--offline", action="store_true")
    p_ing.set_defaults(func=_cmd_ingest_demo)

    p_opt = sub.add_parser("optimise-demo", help="Optimise a synthetic day")
    p_opt.add_argument("--day")
    p_opt.set_defaults(func=_cmd_optimise_demo)

    p_bt = sub.add_parser("backtest", help="Run a benchmark backtest")
    p_bt.add_argument("--days", type=int, default=21)
    p_bt.set_defaults(func=_cmd_backtest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
