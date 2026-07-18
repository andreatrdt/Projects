# Optimisation model

A mixed-integer linear program (MILP) over the horizon of Settlement Periods. Binaries only
enforce that charging and discharging cannot happen at once; all economics are linear, so it
solves quickly with the open-source **HiGHS** solver (optional Gurobi if licensed).

## Units & sign conventions

- Power in **MW**, energy in **MWh**, energy prices in **GBP/MWh**, availability in **GBP/MW/h**.
- `charge_mw`, `discharge_mw` are **grid-side (AC) power**.
- `net_export_mw[t] = discharge_mw[t] − charge_mw[t]` (positive = exporting).
- Objective is a **maximisation** of expected profit; P&L positive = revenue. Charging at a
  negative price is revenue.

## Decision variables (per period *t*)

`charge_mw`, `discharge_mw`, `state_of_charge_mwh`, `charge_mode_binary`,
`discharge_mode_binary`, `upward_service_reserved_mw`, `downward_service_reserved_mw`, and
(portfolio mode) `wholesale_buy_mwh`, `wholesale_sell_mwh`, `residual_imbalance_mwh`.

## Constraints

**No simultaneous charge/discharge**
```
cbin[t] + dbin[t] ≤ 1
0 ≤ charge_mw[t]    ≤ P_charge · cbin[t]
0 ≤ discharge_mw[t] ≤ P_discharge · dbin[t]
```

**State of charge** (Δt derived from timestamps, always 0.5 h)
```
soc[t+1] = soc[t] + η_c · charge_mw[t] · Δt − discharge_mw[t] · Δt / η_d
soc_min ≤ soc[t] ≤ soc_max
soc[0] = initial_soc
soc[N] ≥ minimum_terminal_soc
```

**Power & grid limits**
```
discharge_mw[t] − charge_mw[t] ≤ grid_export_limit
charge_mw[t] − discharge_mw[t] ≤ grid_import_limit
```

**Reserve — conservative simplifications** (documented as such; real DC/DM/DR and Balancing
Reserve rules require more detail)
```
# Power headroom
up[t]   ≤ P_discharge − net_export_mw[t]        # ability to stop charging / discharge more
down[t] ≤ net_export_mw[t] + P_charge           # ability to stop discharging / charge more
# Energy-duration (sustain the service for h hours)
soc[t] − soc_min          ≥ up[t]   · h_up  / η_d
soc_max − soc[t]          ≥ η_c · down[t] · h_down
```

**Ramp limits** (optional) on `net_export` between consecutive periods, and a **daily cycle
limit** measured by discharge throughput.

## Objective

Maximise over *t*:
```
  wholesale_price[t] · (discharge_mwh[t] − charge_mwh[t])           # wholesale
+ up_avail[t]   · up[t]   · Δt                                       # availability (up)
+ down_avail[t] · down[t] · Δt                                       # availability (down)
+ E[BM_up][t]   · up[t]   · Δt + E[BM_down][t] · down[t] · Δt        # expected activation margin
− degradation · (charge_mwh[t] + discharge_mwh[t])                  # degradation
− imbalance_cost[t]                                                 # imbalance exposure (portfolio)
+ terminal_value · soc[N]                                           # terminal battery value
```

Efficiency losses are captured inside the SoC balance (you buy more grid energy than you
store, and remove more stored energy than you deliver), so they are **not** double-counted.
Availability and expected-activation both scale with reserved MW but pay for **different**
things (holding capacity vs being called), so they do not double-count either.

## Service value is decomposed, not a magic price

`gb_battery.optimiser.service.value_service` builds:
```
service value = availability payment + expected activation margin
              − wholesale opportunity cost − energy restoration cost
              − degradation − efficiency losses − expected non-delivery penalty
```
Expected BM value = acceptance_probability × accepted_fraction × activation_margin. Submitting
a price alone earns nothing. Where public data cannot support a reliable estimate, the value is
labelled **estimated**, shown with sensitivity, and never presented as a known future payment.

## Marginal values

- **Per-period value of stored energy** = dual of the SoC-balance constraint. We fix the MILP
  binaries, relax integrality, re-solve the LP and read the duals (a robust "water value").
- **Horizon-level** value of +1 MWh stored / +1 MWh capacity / +1 MW charge / +1 MW discharge
  come from small perturbation resolves. If duals are unavailable, only these are reported.

## Stochastic & CVaR

A single physical schedule is committed (first-stage); profit is evaluated across price
scenarios. Risk aversion penalises downside via the Rockafellar–Uryasev CVaR:
```
maximise  E[profit] − λ · ( η + 1/(1−α) · Σ_s p_s · z_s )
s.t.      z_s ≥ −profit_s − η,   z_s ≥ 0
```
A `robust` mode instead maximises the worst-case scenario. As λ rises, expected P&L falls and
the schedule becomes more conservative.

## Rolling horizon (MPC)

Fetch newest data → update forecasts → optimise the next 24–48 h → commit the first period (or
a configurable near-term block) → update SoC → advance the horizon. Simulated decisions are
never claimed to have been physically executed.
