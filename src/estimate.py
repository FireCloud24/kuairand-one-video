"""Treated-minus-control differences with a bootstrap that resamples users.

Everything is computed from per-user sums, so one bootstrap resample is a weighted
sum over users and 2,000 resamples take a moment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BOOT = 2000
SEED = 20261114


def user_codes(users: np.ndarray) -> tuple[np.ndarray, int]:
    """Map user ids to 0..U-1."""
    uniq, codes = np.unique(np.asarray(users), return_inverse=True)
    return codes, len(uniq)


def bootstrap_weights(n_users: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n_users): how many times each user is drawn in each resample.
    Each row draws n_users users with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=n_boot).astype(np.float64)


def _per_user(values: np.ndarray, codes: np.ndarray, n_users: int) -> np.ndarray:
    """Sum each row of `values` (topics, events) within users -> (topics, users)."""
    return np.vstack([np.bincount(codes, weights=row, minlength=n_users) for row in values])


@dataclass
class Result:
    effect: np.ndarray        # (topics,) treated mean minus control mean
    treated_mean: np.ndarray  # (topics,)
    control_mean: np.ndarray  # (topics,)
    n_treated: np.ndarray     # (topics,) events used
    n_control: np.ndarray
    pooled: float             # average of effect weighted by n_treated
    pooled_treated_mean: float
    pooled_control_mean: float
    boot_effect: np.ndarray   # (n_boot, topics)
    boot_pooled: np.ndarray   # (n_boot,)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        return percentile_interval(self.boot_pooled, level)

    def topic_intervals(self, level: float = 0.95) -> np.ndarray:
        tail = 100 * (1 - level) / 2
        return np.nanpercentile(self.boot_effect, [tail, 100 - tail], axis=0).T


def percentile_interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    lo, hi = np.nanpercentile(draws, [tail, 100 - tail])
    return float(lo), float(hi)


def _pool(effect: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted average over the last axis, skipping topics with no estimate."""
    ok = np.isfinite(effect) & (weight > 0)
    w = np.where(ok, weight, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(ok, effect, 0.0) * w).sum(axis=-1) / w.sum(axis=-1)


def diff_in_means(y: np.ndarray, treated: np.ndarray, codes: np.ndarray, n_users: int,
                  weights: np.ndarray | None = None, keep: np.ndarray | None = None) -> Result:
    """For each topic (row): mean of y over treated events minus mean over control events.

    y        (topics, events), NaN where the event has no value for that topic
    treated  (topics, events) boolean
    codes    (events,) user code of each event
    weights  (n_boot, n_users) from bootstrap_weights; None for no bootstrap
    keep     optional (events,) or (topics, events) boolean; events outside it are dropped
    """
    y = np.asarray(y, dtype=float)
    treated = np.asarray(treated, dtype=bool)
    valid = np.isfinite(y)
    if keep is not None:
        valid &= np.broadcast_to(np.asarray(keep, dtype=bool), y.shape)
    y0 = np.where(valid, y, 0.0)
    t = (treated & valid).astype(float)
    c = (~treated & valid).astype(float)
    s1, n1 = _per_user(y0 * t, codes, n_users), _per_user(t, codes, n_users)
    s0, n0 = _per_user(y0 * c, codes, n_users), _per_user(c, codes, n_users)

    def stats(a1, b1, a0, b0):
        with np.errstate(invalid="ignore", divide="ignore"):
            m1, m0 = a1 / b1, a0 / b0
        return m1, m0, m1 - m0

    S1, N1, S0, N0 = s1.sum(1), n1.sum(1), s0.sum(1), n0.sum(1)
    m1, m0, eff = stats(S1, N1, S0, N0)
    if weights is None:
        boot_eff = np.empty((0, y.shape[0]))
        boot_pooled = np.empty(0)
    else:
        B1 = weights @ n1.T
        _, _, boot_eff = stats(weights @ s1.T, B1, weights @ s0.T, weights @ n0.T)
        boot_pooled = _pool(boot_eff, B1)
    return Result(eff, m1, m0, N1.astype(int), N0.astype(int), float(_pool(eff, N1)),
                  float(_pool(m1, N1)), float(_pool(m0, N1)), boot_eff, boot_pooled)
