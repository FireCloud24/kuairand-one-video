"""The balance table on a synthetic study."""
import numpy as np

from scripts import balance
from src import estimate


def test_standardised_rows_have_unit_sd():
    y = np.array([[1.0, 2.0, 3.0, np.nan], [10.0, 10.0, 30.0, 50.0]])
    z = balance.standardised(y)
    assert np.nanstd(z[0], ddof=1) == 1.0 or abs(np.nanstd(z[0], ddof=1) - 1) < 1e-12
    assert abs(np.nanstd(z[1], ddof=1) - 1) < 1e-12


def test_covariates_are_fixed_before_the_exposure(null_study):
    cov = balance.covariates(null_study)
    assert len(cov) == 5
    for y in cov.values():
        assert y.shape == (2, len(null_study.events))
    base = cov["Baseline share of the topic (last 100 impressions)"]
    assert np.nanmin(base) >= 0 and np.nanmax(base) <= 1


def test_random_exposures_are_balanced(null_study):
    n_users = null_study.events["user_id"].nunique()
    pooled, per_topic = balance.balance_rows(null_study, estimate.bootstrap_weights(n_users, 300, 1))
    base = pooled[0]
    assert base["covariate"].startswith("Baseline")
    assert base["lo"] < 0 < base["hi"]          # chance decided the topic, so no real difference
    assert abs(base["smd"]) < 0.15
    assert [r["topic"] for r in per_topic] == [3, 9]
    md = balance.to_markdown(pooled, per_topic, {"events": 10, "users": 3, "no_before": 1})
    assert "Baseline share by topic" in md and "Games" in md
