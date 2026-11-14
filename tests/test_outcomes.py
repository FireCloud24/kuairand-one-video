"""Outcomes, the decision rule and the event-study script on synthetic studies."""
import numpy as np
import pandas as pd
import pytest

from scripts import event_study
from src import estimate, events, outcomes


def weights_for(study, n_boot=300):
    return estimate.bootstrap_weights(study.events["user_id"].nunique(), n_boot, 7)


def test_outcome_and_baseline_by_hand():
    df = pd.DataFrame({"user_id": [1] * 6, "time_ms": [1, 1, 2, 3, 3, 3],
                       "topic": [3, 9, 3, 3, 9, 9], "tab": [1] * 6})
    feed = events.build_feed(df)
    ev = pd.DataFrame({"user_id": [1], "time_ms": [2], "topic": [3]})
    ev = pd.concat([ev, events.locate(feed, ev["user_id"].to_numpy(), ev["time_ms"].to_numpy())], axis=1)
    study = events.Study(feed, ev, [3, 9], pd.DataFrame())
    assert outcomes.baseline(study)[:, 0].tolist() == [0.5, 0.5]          # rows at t=1
    assert outcomes.outcome(study, 2)[:, 0].tolist() == [0.5, 0.5]        # first two rows at t=3
    assert outcomes.outcome(study, 10)[:, 0].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert study.treated.tolist() == [[True], [False]]


def test_decision_rule():
    m = outcomes.MARGIN
    assert outcomes.decide([(0.001, 0.004), (-0.002, 0.003), (-0.001, 0.001)]) == "the feed bends"
    assert outcomes.decide([(-0.002, 0.003), (-0.004, 0.004), (-0.001, 0.001)]) == "noise"
    assert outcomes.decide([(-0.002, 0.003), (-0.004, m + 0.001), (-0.001, 0.001)]) == "undecided"
    assert outcomes.decide([(-0.02, -0.01), (-0.004, 0.004), (-0.001, 0.001)]) == "the feed moves away"


def test_no_effect_is_found_when_there_is_none(null_study):
    effects = outcomes.window_effects(null_study, weights_for(null_study))
    for k in outcomes.WINDOWS:
        lo, hi = effects[k]["change"].interval(outcomes.DECISION_LEVEL)
        assert lo < 0 < hi


def test_a_planted_effect_is_found_and_fades_with_the_window(effect_study):
    effects = outcomes.window_effects(effect_study, weights_for(effect_study))
    e10, e100 = effects[10]["change"], effects[100]["change"]
    lo, hi = e10.interval(outcomes.DECISION_LEVEL)
    assert lo > 0                      # planted boost of 0.15 over the next 20 impressions
    assert e10.pooled == pytest.approx(0.15, abs=0.04)
    assert e100.pooled < e10.pooled    # 20 boosted impressions diluted over 100


def test_event_bins_show_the_effect_only_after_the_exposure(effect_study):
    bins = outcomes.event_bins(effect_study, weights_for(effect_study))
    assert len(bins) == 2 * outcomes.N_BINS + 1
    assert [b["first"] for b in bins][:2] == [-100, -90] and bins[outcomes.N_BINS]["first"] == 0
    before = [b for b in bins if b["last"] < 0]
    after_near = [b for b in bins if 1 <= b["first"] <= 11]
    after_far = [b for b in bins if b["first"] >= 41]
    assert all(b["lo"] > 0 for b in after_near)
    assert max(abs(b["effect"]) for b in before) < 0.04
    assert max(abs(b["effect"]) for b in after_far) < 0.04


def test_dropout_shares_are_between_zero_and_one(null_study):
    d = outcomes.dropout(null_study, weights_for(null_study))
    assert 0 <= d["overall_left"] <= 1
    lo, hi = d["left"].interval(0.95)
    assert lo < 0 < hi or abs(d["left"].pooled) < 0.05


def test_near_end_limits_the_window(null_study):
    end = outcomes.near_end(null_study)
    ev = null_study.events
    assert (end >= ev["hi"]).all() and (end <= ev["e"]).all()
    # requests are 2 minutes apart, 5 impressions each: at most 30 requests fit in an hour
    assert (end - ev["hi"]).max() <= 150


def test_run_and_markdown(effect_study, tmp_path):
    res = event_study.run(effect_study, weights_for(effect_study))
    assert res["decision"] == "the feed bends"
    assert res["counts"]["events"] == len(effect_study.events)
    md = event_study.to_markdown(res)
    assert "| Next 10 |" in md and "same request" in md and "**the feed bends**" in md
    event_study.figure(res["bins"], tmp_path / "fig.png", event_study.TITLES[res["decision"]])
    assert (tmp_path / "fig.png").stat().st_size > 10_000


def test_tag_membership_reads_comma_lists():
    video_tags = pd.DataFrame({"video_id": [1, 2, 3, 4], "tag": ["39", "12,62", None, "3,39"]})
    member = events.tag_membership(np.array([4, 1, 2, 3, 9]), video_tags, [39, 12, 3])
    assert member.tolist() == [[True, True, False, False, False],
                               [False, False, True, False, False],
                               [True, False, False, False, False]]
