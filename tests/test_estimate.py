"""The difference-in-means estimator and the user bootstrap."""
import numpy as np
import pytest

from src import estimate


def test_user_codes_are_dense():
    codes, n = estimate.user_codes(np.array([50, 7, 50, 9]))
    assert n == 3 and codes.tolist() == [2, 0, 2, 1]


def test_bootstrap_weights_draw_n_users_each_time_and_are_reproducible():
    w = estimate.bootstrap_weights(40, n_boot=25, seed=1)
    assert w.shape == (25, 40)
    assert (w.sum(axis=1) == 40).all()
    assert (w == estimate.bootstrap_weights(40, n_boot=25, seed=1)).all()
    assert not (w == estimate.bootstrap_weights(40, n_boot=25, seed=2)).all()


def test_diff_in_means_by_hand_with_missing_values():
    y = np.array([[1.0, 3.0, 10.0, np.nan, 6.0]])
    treated = np.array([[True, True, False, False, False]])
    codes = np.array([0, 1, 0, 1, 2])
    r = estimate.diff_in_means(y, treated, codes, 3)
    assert r.treated_mean[0] == 2.0 and r.control_mean[0] == 8.0
    assert r.effect[0] == -6.0
    assert r.n_treated[0] == 2 and r.n_control[0] == 2     # the NaN event is dropped
    assert r.pooled == -6.0


def test_pooled_effect_is_weighted_by_treated_events():
    # topic A: 3 treated, effect 1 ; topic B: 1 treated, effect 5 -> pooled (3*1 + 1*5)/4 = 2
    y = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    treated = np.array([[True, True, True, False], [False, False, False, True]])
    r = estimate.diff_in_means(y, treated, np.arange(4), 4)
    assert r.effect.tolist() == [1.0, 5.0]
    assert r.pooled == pytest.approx(2.0)


def test_keep_drops_events():
    y = np.array([[1.0, 9.0, 2.0, 4.0]])
    treated = np.array([[True, True, False, False]])
    r = estimate.diff_in_means(y, treated, np.arange(4), 4, keep=np.array([True, False, True, True]))
    assert r.effect[0] == 1.0 - 3.0


def simulate(effect, n_users=400, per_user=20, seed=0):
    """Users differ a lot in taste; treatment is random and adds `effect`."""
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(n_users), per_user)
    taste = rng.normal(0, 1.0, n_users)[users]
    treated = rng.random(len(users)) < 0.2
    y = taste + rng.normal(0, 1.0, len(users)) + effect * treated
    return y[None, :], treated[None, :], users


def test_recovers_a_known_effect_and_the_interval_covers_it():
    y, treated, users = simulate(0.5)
    codes, n = estimate.user_codes(users)
    r = estimate.diff_in_means(y, treated, codes, n, estimate.bootstrap_weights(n, 500, 3))
    lo, hi = r.interval(0.95)
    assert lo < 0.5 < hi
    assert lo > 0          # a real effect of this size is detected
    assert r.pooled == pytest.approx(0.5, abs=0.1)


def test_a_true_zero_gives_an_interval_around_zero():
    y, treated, users = simulate(0.0, seed=5)
    codes, n = estimate.user_codes(users)
    r = estimate.diff_in_means(y, treated, codes, n, estimate.bootstrap_weights(n, 500, 3))
    lo, hi = r.interval(0.95)
    assert lo < 0 < hi


def test_user_bootstrap_is_wider_than_treating_events_as_independent():
    """When a whole user is treated or not, events inside a user are not independent.
    Resampling users must give a wider interval than the naive standard error."""
    rng = np.random.default_rng(2)
    n_users, per_user = 200, 30
    users = np.repeat(np.arange(n_users), per_user)
    treated_user = rng.random(n_users) < 0.5
    treated = treated_user[users]
    y = rng.normal(0, 1.0, n_users)[users] + rng.normal(0, 0.2, len(users))
    codes, n = estimate.user_codes(users)
    r = estimate.diff_in_means(y[None, :], treated[None, :], codes, n, estimate.bootstrap_weights(n, 500, 4))
    lo, hi = r.interval(0.95)
    naive_se = np.sqrt(y[treated].var(ddof=1) / treated.sum() + y[~treated].var(ddof=1) / (~treated).sum())
    assert (hi - lo) > 3 * (2 * 1.96 * naive_se)


def test_percentile_interval():
    lo, hi = estimate.percentile_interval(np.arange(101.0), 0.90)
    assert (lo, hi) == pytest.approx((5.0, 95.0))
