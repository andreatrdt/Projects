"""Generate concise, input-derived trader explanations for each decision.

The text is assembled from the actual decision, the binding constraints and the
period's economic drivers — never from generic boilerplate. If nothing is binding
and the action is IDLE, the explanation says *why* (e.g. spread below round-trip
cost).
"""

from __future__ import annotations

from gb_battery.optimiser.results import ActionLabel, PeriodResult


def _fmt(x: float | None, unit: str = "", dp: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.{dp}f}{unit}"


def explain_period(
    r: PeriodResult,
    max_charge_mw: float,
    max_discharge_mw: float,
    round_trip_efficiency: float,
) -> str:
    """Return a one-to-three sentence explanation for a single period."""
    price = r.wholesale_price
    parts: list[str] = []
    binding = set(r.binding_constraints)

    if r.action == ActionLabel.CHARGE:
        why_price = "the wholesale price is negative" if price < 0 else (
            f"the wholesale price ({_fmt(price, ' GBP/MWh', 0)}) is low relative to later periods"
        )
        lead = f"Charge at {_fmt(r.charge_mw, ' MW')} during SP{r.settlement_period:02d} because {why_price}"
        if "charge_power_max" in binding:
            lead += ", running at the full charge-power limit"
        elif "soc_max" in binding or "battery_full" in binding:
            lead += "; charging stops as the battery reaches its usable maximum"
        parts.append(lead + ".")
    elif r.action == ActionLabel.DISCHARGE:
        lead = (
            f"Discharge {_fmt(r.discharge_mw, ' MW')} during SP{r.settlement_period:02d} "
            f"because the wholesale price ({_fmt(price, ' GBP/MWh', 0)}) is high"
        )
        if "discharge_power_max" in binding:
            lead += " and the discharge-power limit binds"
        elif r.upward_reserved_mw > 1e-6:
            lead += (
                f", but only {_fmt(r.discharge_mw, ' MW')} rather than the "
                f"{_fmt(max_discharge_mw, ' MW')} maximum because "
                f"{_fmt(r.upward_reserved_mw, ' MW')} is reserved for upward response"
            )
        parts.append(lead + ".")
    elif r.action == ActionLabel.RESERVE_UP:
        parts.append(
            f"Hold {_fmt(r.upward_reserved_mw, ' MW')} of upward reserve in SP{r.settlement_period:02d}: "
            f"the availability value exceeds the arbitrage value at the current price "
            f"({_fmt(price, ' GBP/MWh', 0)})."
        )
    elif r.action == ActionLabel.RESERVE_DOWN:
        parts.append(
            f"Hold {_fmt(r.downward_reserved_mw, ' MW')} of downward reserve in SP{r.settlement_period:02d}: "
            f"keeping empty capacity is worth more than charging now."
        )
    else:  # IDLE
        parts.append(
            f"Stay idle in SP{r.settlement_period:02d}: the price ({_fmt(price, ' GBP/MWh', 0)}) "
            f"does not clear the round-trip efficiency loss "
            f"({round_trip_efficiency*100:.0f}% RTE) plus degradation cost."
        )

    # Add binding-constraint colour where informative.
    if "terminal_soc" in binding:
        parts.append("The terminal state-of-charge requirement is binding this period.")
    if "up_energy" in binding:
        parts.append("Upward reserve is energy-limited: stored energy caps the reservable MW.")
    if "down_energy" in binding:
        parts.append("Downward reserve is energy-limited: empty headroom caps the reservable MW.")
    if "cycle_limit" in binding:
        parts.append("The daily cycling limit is binding across the horizon.")

    # Marginal-value insight.
    mv = r.marginals
    if (
        mv.stored_energy_gbp_per_mwh is not None
        and r.action == ActionLabel.DISCHARGE
        and mv.stored_energy_gbp_per_mwh > price
    ):
        parts.append(
            f"Stored energy is valued at {_fmt(mv.stored_energy_gbp_per_mwh, ' GBP/MWh', 1)} "
            "internally, so further discharge is held back for higher-value periods."
        )
    return " ".join(parts)
