import numpy as np
import pandas as pd

from Big_Levy_Investments.PortfolioReplica.ReplicaClass import PortfolioReplicator


def make_replicator(transaction_cost=0.0, target_override=None):
    rng = np.random.default_rng(7)
    n_obs, n_assets = 80, 3
    x = rng.normal(0.0, 0.015, size=(n_obs, n_assets))
    beta = np.array([0.55, -0.20, 0.35])
    y = x @ beta + rng.normal(0.0, 0.002, size=n_obs)
    if target_override is not None:
        y = target_override(y.copy())

    index = pd.date_range("2020-01-03", periods=n_obs, freq="W-FRI")
    replica = object.__new__(PortfolioReplicator)
    replica.underlyings_returns = pd.DataFrame(
        x, index=index, columns=[f"asset_{i}" for i in range(n_assets)]
    )
    replica.target_returns = pd.DataFrame({"target": y}, index=index)
    replica.transaction_cost_rate = transaction_cost
    replica.annual_factor = 52
    replica.var_confidence = 0.95
    replica.var_horizon = 1
    replica.max_var_threshold = 0.05
    replica._run_name = None
    return replica


def test_var_is_a_positive_loss_number():
    replica = make_replicator()
    value = replica.calculate_var([-0.02, 0.01, -0.01, 0.03])
    assert value > 0.0


def test_kalman_observation_cannot_change_same_period_return():
    base = make_replicator()
    shocked = make_replicator(
        target_override=lambda y: np.concatenate([y[:20], [1.0], y[21:]])
    )

    base_result = base.run_kalman_filter_model(20, 1)
    shocked_result = shocked.run_kalman_filter_model(20, 1)

    assert base_result["replica_returns"].iloc[0] == shocked_result["replica_returns"].iloc[0]
    assert base_result["replica_returns"].iloc[1] != shocked_result["replica_returns"].iloc[1]


def test_enkf_is_reproducible_and_has_no_same_period_lookahead():
    base = make_replicator()
    shocked = make_replicator(
        target_override=lambda y: np.concatenate([y[:20], [1.0], y[21:]])
    )

    result_a = base.run_ensemble_kalman_filter_model(20, 1, 30, 0.01, random_state=11)
    result_b = make_replicator().run_ensemble_kalman_filter_model(
        20, 1, 30, 0.01, random_state=11
    )
    shocked_result = shocked.run_ensemble_kalman_filter_model(
        20, 1, 30, 0.01, random_state=11
    )

    np.testing.assert_allclose(result_a["replica_returns"], result_b["replica_returns"])
    assert result_a["replica_returns"].iloc[0] == shocked_result["replica_returns"].iloc[0]
    assert result_a["replica_returns"].iloc[1] != shocked_result["replica_returns"].iloc[1]


def test_filter_uses_observations_between_trading_dates():
    base = make_replicator().run_kalman_filter_model(20, 3)
    shocked = make_replicator(
        target_override=lambda y: np.concatenate([y[:20], [1.0], y[21:]])
    ).run_kalman_filter_model(20, 3)

    np.testing.assert_allclose(
        base["replica_returns"].iloc[:3], shocked["replica_returns"].iloc[:3]
    )
    assert base["replica_returns"].iloc[3] != shocked["replica_returns"].iloc[3]


def test_ensemble_size_is_used_by_the_filter():
    small = make_replicator().run_ensemble_kalman_filter_model(
        20, 1, ensemble_size=10, process_noise_scale=0.01, random_state=5
    )
    large = make_replicator().run_ensemble_kalman_filter_model(
        20, 1, ensemble_size=80, process_noise_scale=0.01, random_state=5
    )

    assert not np.allclose(small["replica_returns"], large["replica_returns"])


def test_transaction_cost_is_subtracted_once_on_each_execution_date():
    gross = make_replicator(transaction_cost=0.0).run_kalman_filter_model(20, 3)
    net = make_replicator(transaction_cost=0.002).run_kalman_filter_model(20, 3)

    return_difference = gross["replica_returns"] - net["replica_returns"]
    np.testing.assert_allclose(return_difference, net["transaction_costs"], atol=1e-14)
    assert net["transaction_costs"][0] == 0.0
    assert net["transaction_costs"][1] == 0.0
    assert net["transaction_costs"][2] == 0.0
    assert net["transaction_costs"][3] > 0.0


def test_elasticnet_transaction_cost_is_not_carried_forward():
    gross = make_replicator(transaction_cost=0.0).run_elasticnet_normalized(
        l1_ratio=0.5, rolling_window=20, alpha=0.01, rebalancing_window=3
    )
    net = make_replicator(transaction_cost=0.002).run_elasticnet_normalized(
        l1_ratio=0.5, rolling_window=20, alpha=0.01, rebalancing_window=3
    )

    return_difference = gross["replica_returns"] - net["replica_returns"]
    np.testing.assert_allclose(return_difference, net["transaction_costs"], atol=1e-14)
    assert net["transaction_costs"][1] == 0.0
    assert net["transaction_costs"][2] == 0.0


def test_var_scaling_respects_the_limit_using_past_data():
    replica = make_replicator()
    replica.max_var_threshold = 0.005
    result = replica.run_kalman_filter_model(20, 2)

    finite_var = np.asarray(result["var_values"])[
        np.isfinite(result["var_values"])
    ]
    assert finite_var.size > 0
    assert finite_var.max() <= replica.max_var_threshold + 1e-12


def test_price_based_return_helper_subtracts_reported_costs():
    replica = make_replicator()
    index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    prices = pd.DataFrame(
        {"a": [100.0, 110.0, 121.0], "b": [100.0, 100.0, 100.0]},
        index=index,
    )
    weights = pd.DataFrame(
        {"a": [1.0, 0.5, 0.5], "b": [0.0, 0.5, 0.5]},
        index=index,
    )

    returns, total_cost, _ = replica.compute_replica_returns_from_weights_and_prices(
        weights, prices, tc_rate=0.01
    )

    assert total_cost == 0.01
    assert returns.iloc[1] == 0.04  # 0.5 * 10% gross minus 1% turnover cost
    assert returns.iloc[2] == 0.05
