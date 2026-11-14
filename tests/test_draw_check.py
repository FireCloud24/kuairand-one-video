"""Counting repeated random videos and random videos already seen in the feed."""
import pandas as pd
import pytest

from scripts import draw_check


def test_repeat_stats_counts_and_expectation():
    random = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "video_id": [10, 11, 10, 10, 12], "time_ms": [1, 2, 3, 1, 2]})
    r = draw_check.repeat_stats(random)
    assert r["pool"] == 3 and r["exposures"] == 5
    assert r["repeats_observed"] == 1                      # user 1 got video 10 twice
    # user 1: 3 draws -> 3 pairs; user 2: 2 draws -> 1 pair; each pair matches with chance 1/3
    assert r["repeats_expected_with_replacement"] == pytest.approx(4 / 3)


def test_seen_before_counts_only_earlier_showings():
    random = pd.DataFrame({"user_id": [1, 1, 2], "video_id": [10, 11, 10], "time_ms": [5, 5, 5]})
    standard = pd.DataFrame({"user_id": [1, 1, 2, 2], "video_id": [10, 11, 10, 99], "time_ms": [1, 9, 9, 1]})
    r = draw_check.seen_before_stats(random, standard)
    assert r["already_in_feed_observed"] == 1              # only user 1, video 10 came before
    # pool = {10, 11}. User 1 had seen 1 pool video before t=5 for each of 2 exposures -> 2 * 1/2.
    # User 2 had seen none before t=5.
    assert r["already_in_feed_expected_if_ignored"] == pytest.approx(1.0)


def test_markdown_has_both_rows():
    md = draw_check.to_markdown({"pool": 3, "exposures": 5, "repeats_observed": 1,
                                 "repeats_expected_with_replacement": 1.3, "already_in_feed_observed": 1,
                                 "already_in_feed_expected_if_ignored": 1.0})
    assert "shown twice" in md and "ordinary feed" in md
