"""Subgroup masks and the heterogeneity table on a synthetic study."""
import numpy as np

from scripts import heterogeneity
from src import estimate


def test_subgroup_masks_split_the_events(null_study, monkeypatch):
    monkeypatch.setattr(heterogeneity, "POPULAR", (3,))
    groups = {g["label"]: g for g in heterogeneity.subgroups(null_study)}
    n_topics, n_events = len(null_study.targets), len(null_study.events)
    for g in groups.values():
        assert g["keep"].shape == (n_topics, n_events)
    new = groups[f"New: registered {heterogeneity.NEW_MAX_DAYS} days or fewer"]["keep"]
    old = groups["Registered longer"]["keep"]
    assert (new ^ old).all()                                  # every event is in exactly one
    watched = groups["Random video watched"]["keep"]
    skipped = groups["Random video skipped"]["keep"]
    assert (watched ^ skipped).all()
    assert watched[0].sum() == (null_study.events["is_click"] == 1).sum()
    popular, niche = groups["Popular topics (6)"]["keep"], groups["Niche topics (7)"]["keep"]
    assert popular[0].all() and not popular[1].any()          # topic 3 is "popular" here
    assert niche[1].all() and not niche[0].any()
    heavy = groups["Heavy: top third by impressions before"]["keep"]
    light = groups["Light: bottom third by impressions before"]["keep"]
    assert not (heavy & light).any()


def test_new_users_mask_follows_register_days(null_study):
    groups = {g["label"]: g for g in heterogeneity.subgroups(null_study)}
    new = groups[f"New: registered {heterogeneity.NEW_MAX_DAYS} days or fewer"]["keep"][0]
    reg = null_study.events["user_id"].map(null_study.users.set_index("user_id")["register_days"]).to_numpy()
    assert (new == (reg <= heterogeneity.NEW_MAX_DAYS)).all()


def test_run_finds_the_planted_effect_in_each_topic_group(effect_study, monkeypatch, tmp_path):
    monkeypatch.setattr(heterogeneity, "POPULAR", (3,))
    weights = estimate.bootstrap_weights(effect_study.events["user_id"].nunique(), 200, 5)
    rows = heterogeneity.run(effect_study, weights)
    by = {r["label"]: r for r in rows}
    assert by["All exposures"]["lo_10"] > 0
    assert by["Popular topics (6)"]["effect_10"] > 0.05
    assert by["Niche topics (7)"]["effect_10"] > 0.05
    # subgroup counts add up to the whole
    assert by["Random video watched"]["n_10"] + by["Random video skipped"]["n_10"] == by["All exposures"]["n_10"]
    assert by["Popular topics (6)"]["n_10"] + by["Niche topics (7)"]["n_10"] == by["All exposures"]["n_10"]
    md = heterogeneity.to_markdown(rows, heterogeneity.watch_rates(effect_study))
    assert "Random video skipped" in md and "do not" in md
    heterogeneity.figure(rows, tmp_path / "h.png")
    assert (tmp_path / "h.png").stat().st_size > 10_000


def test_watch_rates(null_study):
    r = heterogeneity.watch_rates(null_study)
    assert 0 < r["long_view"] < r["watched"] < 1 or np.isclose(r["watched"], r["long_view"])


def test_watching_bins_and_summary(effect_study, tmp_path):
    weights = estimate.bootstrap_weights(effect_study.events["user_id"].nunique(), 100, 5)
    bins = heterogeneity.watching_bins(effect_study, weights)
    assert set(bins) == {"watched", "skipped"}
    n = {k: v[0]["n_treated"] for k, v in bins.items()}
    assert n["watched"] < n["skipped"]                 # 30% of synthetic random videos are watched
    summary = heterogeneity.bin_summary(bins["skipped"])
    # the planted effect lasts 20 impressions, so the 3 bins after sit above the bins before
    assert summary["rise"] > 0.05
    assert summary["rise"] == summary["after_30"] - summary["before"]
    md = heterogeneity.to_markdown([], {"watched": 0.3, "long_view": 0.1}, bins)
    assert "| Watched |" in md and "| Skipped |" in md
    heterogeneity.watching_figure(bins, tmp_path / "w.png")
    assert (tmp_path / "w.png").stat().st_size > 10_000


def test_bin_summary_by_hand():
    bins = [{"first": -20, "last": -11, "effect": 0.01}, {"first": -10, "last": -1, "effect": 0.03},
            {"first": 0, "last": 0, "effect": 0.5},
            {"first": 1, "last": 10, "effect": 0.05}, {"first": 11, "last": 20, "effect": 0.07},
            {"first": 21, "last": 30, "effect": 0.03}, {"first": 31, "last": 40, "effect": 9.0}]
    s = heterogeneity.bin_summary(bins)
    assert np.isclose(s["before"], 0.02) and s["last_before"] == 0.03
    assert np.isclose(s["after_30"], 0.05) and np.isclose(s["rise"], 0.03)
