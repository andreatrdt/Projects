"""Balancing Mechanism acceptance research module (exploratory).

Goal: estimate, for a submitted Bid or Offer, the probability it is accepted and the
expected accepted volume — because submitting a price alone earns nothing.

Method:
1. Link BOD submissions to BOALF acceptances by (settlement date, period, BM unit).
2. Label each BOD pair as accepted if an overlapping BOALF action moved the unit into
   that pair's level band (an approximate, documented heuristic — exact pairId↔BOALF
   attribution is not always recoverable from public schemas).
3. Build **leakage-safe** features known at submission time (price distance from a
   reference, settlement period, level band width) — never final system prices.
4. Fit a probability model.

IMPORTANT (per project scope): the production optimiser does **not** depend on this
module. Where matching is unreliable, use user-defined or historical service-value
assumptions and clearly label estimates. This module surfaces caveats explicitly via
:func:`exploratory_report`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def link_bod_boalf(bod: pd.DataFrame, boalf: pd.DataFrame) -> pd.DataFrame:
    """Label each BOD pair as accepted/not using overlapping BOALF acceptances.

    Heuristic: for a given (date, period, bm_unit), an *offer* pair (positive level
    band) is treated as accepted if a BOALF action for that unit reaches into the
    pair's [level_from, level_to] band; symmetrically for *bid* pairs (negative band).
    """
    if bod.empty:
        return bod.assign(accepted=pd.Series(dtype=int), accepted_level=pd.Series(dtype=float))

    # Index BOALF acceptances by unit & period for quick lookup.
    acc = boalf.copy()
    labelled = []
    for _, r in bod.iterrows():
        unit = r.get("bm_unit")
        sp = r.get("settlement_period")
        lo = min(_f(r.get("level_from")), _f(r.get("level_to")))
        hi = max(_f(r.get("level_from")), _f(r.get("level_to")))
        cand = acc[(acc.get("bm_unit") == unit)]
        if "settlement_period_from" in cand.columns:
            cand = cand[(cand["settlement_period_from"] <= sp) & (cand["settlement_period_to"] >= sp)]
        accepted = 0
        accepted_level = 0.0
        for _, a in cand.iterrows():
            a_hi = max(_f(a.get("level_from")), _f(a.get("level_to")))
            a_lo = min(_f(a.get("level_from")), _f(a.get("level_to")))
            # Overlap between accepted band and the pair's band.
            overlap = min(hi, a_hi) - max(lo, a_lo)
            if overlap > 1e-6:
                accepted = 1
                accepted_level = max(accepted_level, overlap)
        row = r.to_dict()
        row["accepted"] = accepted
        row["accepted_level"] = accepted_level
        labelled.append(row)
    return pd.DataFrame(labelled)


ACCEPT_FEATURES = ["settlement_period", "is_offer", "band_width", "price_distance"]


def build_acceptance_features(labelled: pd.DataFrame, reference_price: float | None = None) -> pd.DataFrame:
    """Add leakage-safe features for the acceptance model.

    ``reference_price`` is a submission-time market reference (e.g. recent MID). The
    feature ``price_distance`` is how far the offer/bid price sits from it.
    """
    df = labelled.copy()
    df["is_offer"] = (df["offer_price"].notna() & (df.get("pair_id", 0) >= 0)).astype(int)
    df["submitted_price"] = np.where(df["is_offer"] == 1, df["offer_price"], df["bid_price"])
    df["band_width"] = (df["level_to"].astype(float) - df["level_from"].astype(float)).abs()
    ref = reference_price if reference_price is not None else float(np.nanmedian(df["submitted_price"]))
    df["price_distance"] = df["submitted_price"].astype(float) - ref
    return df.dropna(subset=["submitted_price"])


@dataclass
class AcceptanceModel:
    """Logistic acceptance-probability model with an accepted-volume regressor."""

    clf: Any
    reg: Any | None
    n_train: int
    base_rate: float

    def predict_acceptance_probability(self, features: pd.DataFrame) -> np.ndarray:
        x = features[ACCEPT_FEATURES].astype(float)
        return self.clf.predict_proba(x)[:, 1]

    def expected_accepted_volume(self, features: pd.DataFrame) -> np.ndarray:
        prob = self.predict_acceptance_probability(features)
        if self.reg is None:
            return prob * features["band_width"].to_numpy(dtype=float)
        vol = self.reg.predict(features[ACCEPT_FEATURES].astype(float))
        return prob * np.clip(vol, 0, None)


def fit_acceptance_model(features: pd.DataFrame, random_state: int = 0) -> AcceptanceModel:
    """Fit acceptance probability (+ accepted volume) models chronologically-safe."""
    from sklearn.linear_model import LogisticRegression

    x = features[ACCEPT_FEATURES].astype(float)
    y = features["accepted"].astype(int)
    base_rate = float(y.mean()) if len(y) else 0.0
    if y.nunique() < 2:
        # Degenerate: cannot fit a classifier; return a constant-rate stub.
        clf = _ConstantClassifier(base_rate)
        return AcceptanceModel(clf=clf, reg=None, n_train=len(y), base_rate=base_rate)
    clf = LogisticRegression(max_iter=500, random_state=random_state).fit(x, y)
    reg = None
    accepted = features[features["accepted"] == 1]
    if len(accepted) >= 10:
        from sklearn.linear_model import LinearRegression

        reg = LinearRegression().fit(accepted[ACCEPT_FEATURES].astype(float), accepted["accepted_level"].astype(float))
    return AcceptanceModel(clf=clf, reg=reg, n_train=len(y), base_rate=base_rate)


def exploratory_report(bod: pd.DataFrame, boalf: pd.DataFrame) -> dict:
    """Summary + honest caveats for the BM acceptance analysis."""
    labelled = link_bod_boalf(bod, boalf)
    n = len(labelled)
    rate = float(labelled["accepted"].mean()) if n else 0.0
    return {
        "n_bod_pairs": int(n),
        "n_boalf_actions": int(len(boalf)),
        "empirical_acceptance_rate": round(rate, 4),
        "caveats": [
            "pairId-to-BOALF attribution is approximate; overlap heuristic may over/under-count.",
            "Public schemas do not always expose the exact accepted pair; treat as ESTIMATED.",
            "The production optimiser uses user/historical service-value assumptions, not this model.",
            "No final system prices or later revisions are used as features (leakage-safe).",
        ],
    }


class _ConstantClassifier:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, x) -> np.ndarray:
        n = len(x)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
