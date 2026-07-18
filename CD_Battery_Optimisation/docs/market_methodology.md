# GB market methodology

## Settlement Periods

The GB market settles in half-hourly **Settlement Periods (SPs)**. A **Settlement Date** runs
from local (Europe/London) midnight to the next. Because of British Summer Time:

- **46 SPs** on the spring-forward day (clocks 01:00→02:00; one hour skipped)
- **48 SPs** on a normal day
- **50 SPs** on the autumn-back day (clocks 02:00→01:00; one hour repeated)

Each SP is always **30 real minutes**, so per-period duration Δt = 0.5 h, but the number of
periods per day varies. Code derives duration from timestamps and never assumes 48 SPs
(`gb_battery.settlement`).

## Wholesale vs Balancing Mechanism vs imbalance

- **Wholesale** — day-ahead and intraday energy trading. This project uses Elexon **Market
  Index Data (MID)** as a short-term reference price. MID is *not* a full EPEX order book and
  does not represent achievable execution or depth.
- **Balancing Mechanism (BM)** — after gate closure NESO balances the system by accepting
  **Bids** (reduce output / increase demand) and **Offers** (increase output / reduce demand)
  submitted as **Bid-Offer Data (BOD)**. Acceptances appear as **BOALF** (acceptance levels)
  and **BOAV** (volumes). Submitting a bid/offer price alone earns nothing — revenue depends on
  being **accepted**.
- **Imbalance settlement** — any residual difference between contracted and metered position is
  settled at the **system price** (single-price regime). Being short pays the system price;
  being long receives it. `netImbalanceVolume` and the system price indicate whether the system
  is long or short.

## System long/short

We derive a `prob_short` indicator from residual demand (demand − wind − solar) and, where
available, the net imbalance volume. A tight (short) system tends to raise imbalance prices and
the value of upward flexibility; a long system raises the value of downward flexibility.

## Ancillary services

NESO procures frequency response and reserve products — **Dynamic Containment (DC)**,
**Dynamic Moderation (DM)**, **Dynamic Regulation (DR)**, **Balancing Reserve**, and Quick/Slow
Reserve — plus capacity via the **Enduring Auction Capability (EAC)**. Batteries earn an
**availability** payment for holding capability and may earn an **activation** margin when
called. Real product rules (symmetry, baselining, duration, response speed, delivery
verification, non-delivery penalties) are more detailed than the conservative headroom
constraints modelled here.

## Fundamentals used

National demand (forecast & outturn), wind & solar (forecast & outturn), residual demand,
generation mix by fuel, and interconnector flows. These drive both the display and the
forecasting features.
