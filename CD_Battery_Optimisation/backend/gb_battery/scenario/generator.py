"""Scenario generation for stochastic optimisation.

Turns a per-period quantile price forecast into a set of correlated price paths plus
matching system-direction / imbalance-price scenarios. Correlation across settlement
periods is induced by an AR(1) process on the sampling percentile, so a "high-price
day" scenario is high across the day rather than independently noisy per period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gb_battery.optimiser.inputs import OptimisationInputs


@dataclass
class ScenarioSet:
    """A weighted set of price (and imbalance) scenarios aligned to inputs.periods."""

    wholesale_prices: np.ndarray  # shape [S, T]
    imbalance_prices: np.ndarray  # shape [S, T]
    probabilities: np.ndarray  # shape [S], sums to 1

    @property
    def n_scenarios(self) -> int:
        return self.wholesale_prices.shape[0]

    @property
    def horizon(self) -> int:
        return self.wholesale_prices.shape[1]

    def expected_prices(self) -> np.ndarray:
        return np.average(self.wholesale_prices, axis=0, weights=self.probabilities)


def _interp_quantiles(percentile: float, q_levels: np.ndarray, q_values: np.ndarray) -> float:
    """Piecewise-linear inverse-CDF interpolation from quantile pairs."""
    return float(np.interp(percentile, q_levels, q_values))


def generate_price_scenarios(
    inputs: OptimisationInputs,
    *,
    n_scenarios: int = 20,
    sigma_floor: float = 3.0,
    ar_rho: float = 0.8,
    imbalance_spread: float = 15.0,
    seed: int = 0,
) -> ScenarioSet:
    """Generate correlated price/imbalance scenarios from per-period forecasts.

    Uses each period's ``wholesale_price`` (mean) and ``wholesale_price_sigma`` to form
    a normal quantile band, then samples percentiles via an AR(1) process for temporal
    correlation. Equal probabilities are assigned to sampled scenarios.
    """
    rng = np.random.default_rng(seed)
    periods = inputs.periods
    t = len(periods)
    mean = np.array([p.wholesale_price for p in periods])
    sigma = np.array([max(p.wholesale_price_sigma, sigma_floor) for p in periods])

    prices = np.zeros((n_scenarios, t))
    imb = np.zeros((n_scenarios, t))

    for s in range(n_scenarios):
        # AR(1) on the standard-normal shock across periods.
        z = np.zeros(t)
        z[0] = rng.standard_normal()
        for i in range(1, t):
            z[i] = ar_rho * z[i - 1] + np.sqrt(1 - ar_rho**2) * rng.standard_normal()
        prices[s] = mean + sigma * z
        # Imbalance price: wider spread, system short when price high.
        short_bias = np.tanh((prices[s] - mean) / (sigma + 1e-9))
        imb[s] = prices[s] + imbalance_spread * short_bias + rng.standard_normal(t) * sigma * 0.3

    probs = np.full(n_scenarios, 1.0 / n_scenarios)
    return ScenarioSet(prices, imb, probs)


def scenarios_from_inputs_expected(inputs: OptimisationInputs) -> ScenarioSet:
    """Degenerate single-scenario set equal to the deterministic expected case."""
    mean = np.array([[p.wholesale_price for p in inputs.periods]])
    imb = np.array([[(p.expected_imbalance_price or p.wholesale_price) for p in inputs.periods]])
    return ScenarioSet(mean, imb, np.array([1.0]))
