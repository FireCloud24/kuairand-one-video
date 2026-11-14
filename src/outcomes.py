"""Outcomes after a random exposure, as pre-registered.

Outcome for topic T and window k: the share of the next k ordinary impressions in T.
Baseline: the share of T in the last 100 ordinary impressions before the exposure.
Effect: mean of (outcome - baseline) over treated minus the same over control.
"""
from __future__ import annotations

import numpy as np

from . import estimate, events

WINDOWS = (10, 30, 100)
BIN = 10
N_BINS = 10
DECISION_LEVEL = 1 - 0.05 / 3   # three pooled tests share a 5% error rate
MARGIN = 0.005                  # half a percentage point
LEFT_MINUTES = 30
NEAR_MINUTES = 60


def baseline(study: events.Study) -> np.ndarray:
    a, b = events.before_window(study.events, events.BASELINE_N)
    return study.shares(a, b)


def outcome(study: events.Study, k: int, end: np.ndarray | None = None) -> np.ndarray:
    a, b = events.after_window(study.events, k, end=end)
    return study.shares(a, b)


def window_effects(study: events.Study, weights: np.ndarray | None, keep: np.ndarray | None = None,
                   end: np.ndarray | None = None, windows=WINDOWS) -> dict:
    """For each window: the pre-registered estimate ("change", outcome minus baseline)
    and the plain difference in outcomes ("plain")."""
    study.prepare()
    codes, n_users = estimate.user_codes(study.events["user_id"].to_numpy())
    base = baseline(study)
    out = {}
    for k in windows:
        y = outcome(study, k, end)
        out[k] = {
            "change": estimate.diff_in_means(y - base, study.treated, codes, n_users, weights, keep),
            "plain": estimate.diff_in_means(y, study.treated, codes, n_users, weights, keep),
        }
    return out


def decide(intervals: list[tuple[float, float]], margin: float = MARGIN) -> str:
    """The pre-registered rule, applied to the three pooled decision intervals."""
    if any(lo > 0 for lo, hi in intervals):
        return "the feed bends"
    if all(-margin < lo and hi < margin for lo, hi in intervals):
        return "noise"
    if any(hi < 0 for lo, hi in intervals):
        return "the feed moves away"
    return "undecided"


def event_bins(study: events.Study, weights: np.ndarray | None, keep: np.ndarray | None = None) -> list[dict]:
    """Treated minus control share of the topic in bins of 10 impressions around the
    exposure, with no baseline subtracted. Position 0 is the same feed request."""
    study.prepare()
    ev = study.events
    codes, n_users = estimate.user_codes(ev["user_id"].to_numpy())
    windows = []
    for i in range(N_BINS, 0, -1):          # -100..-91, ..., -10..-1
        a, b = events.before_window(ev, BIN, offset=(i - 1) * BIN)
        windows.append((-i * BIN, -(i - 1) * BIN - 1, a, b))
    windows.append((0, 0, ev["lo"].to_numpy(), ev["hi"].to_numpy()))
    for i in range(N_BINS):                 # +1..+10, ..., +91..+100
        a, b = events.after_window(ev, BIN, offset=i * BIN)
        windows.append((i * BIN + 1, (i + 1) * BIN, a, b))
    rows = []
    for first, last, a, b in windows:
        r = estimate.diff_in_means(study.shares(a, b), study.treated, codes, n_users, weights, keep)
        lo, hi = r.interval(0.95) if weights is not None else (np.nan, np.nan)
        rows.append({"first": first, "last": last, "effect": r.pooled, "lo": lo, "hi": hi,
                     "treated_mean": r.pooled_treated_mean, "control_mean": r.pooled_control_mean,
                     "n_treated": int(r.n_treated.sum())})
    return rows


def dropout(study: events.Study, weights: np.ndarray | None) -> dict:
    """Treated against control: did the user leave, and is there any later impression?"""
    study.prepare()
    ev = study.events
    codes, n_users = estimate.user_codes(ev["user_id"].to_numpy())
    n_topics = len(study.targets)
    left = events.left_within(study.feed, ev, ev["time_ms"].to_numpy(), LEFT_MINUTES).astype(float)
    none_after = (ev["hi"] == ev["e"]).to_numpy().astype(float)
    out = {}
    for name, y in (("left", left), ("none_after", none_after)):
        out[name] = estimate.diff_in_means(np.broadcast_to(y, (n_topics, len(y))), study.treated,
                                           codes, n_users, weights)
    out["overall_left"] = float(left.mean())
    out["overall_none_after"] = float(none_after.mean())
    return out


def near_end(study: events.Study) -> np.ndarray:
    """End positions that keep only impressions within NEAR_MINUTES of the exposure."""
    ev = study.events
    return events.end_within(study.feed, ev, ev["time_ms"].to_numpy(), NEAR_MINUTES)
