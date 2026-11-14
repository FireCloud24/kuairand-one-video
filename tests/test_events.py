"""Placing random exposures in the feed and measuring windows around them."""
import numpy as np
import pandas as pd
import pytest

from src import events


def small_feed():
    # user 1: requests at t=10 (3 rows), t=20 (2 rows), t=30 (3 rows); user 2: t=5 (2 rows)
    df = pd.DataFrame({
        "user_id": [2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        "time_ms": [5, 10, 10, 10, 20, 20, 30, 30, 30, 5],
        "topic":   [7, 3, 3, 9, 3, 9, 9, 3, 3, 7],
        "tab":     [1] * 10,
    })
    return events.build_feed(df)


def test_build_feed_sorts_by_user_then_time_keeping_file_order():
    feed = small_feed()
    assert feed.user.tolist() == [1] * 8 + [2] * 2
    assert feed.time.tolist() == [10, 10, 10, 20, 20, 30, 30, 30, 5, 5]
    assert feed.topic.tolist() == [3, 3, 9, 3, 9, 9, 3, 3, 7, 7]


def test_locate_splits_before_same_request_and_after():
    feed = small_feed()
    loc = events.locate(feed, np.array([1, 1, 1, 2, 3]), np.array([20, 25, 5, 5, 1]))
    # exposure in the t=20 request: 3 before, 2 in the same request, 3 after
    assert loc.iloc[0].tolist() == [0, 3, 5, 8]
    # exposure between requests: nothing shares its timestamp
    assert loc.iloc[1].tolist() == [0, 5, 5, 8]
    # before everything
    assert loc.iloc[2].tolist() == [0, 0, 0, 8]
    # user 2: the only request is the same request, so nothing before or after
    assert loc.iloc[3].tolist() == [8, 8, 10, 10]
    # unknown user: empty
    assert loc.iloc[4]["s"] == loc.iloc[4]["e"]


def test_windows_exclude_the_same_request_and_stop_at_the_users_edges():
    feed = small_feed()
    loc = events.locate(feed, np.array([1]), np.array([20]))
    assert [x.tolist() for x in events.before_window(loc, 2)] == [[1], [3]]
    assert [x.tolist() for x in events.before_window(loc, 100)] == [[0], [3]]
    assert [x.tolist() for x in events.after_window(loc, 2)] == [[5], [7]]
    assert [x.tolist() for x in events.after_window(loc, 100)] == [[5], [8]]   # never into user 2
    assert [x.tolist() for x in events.after_window(loc, 2, offset=2)] == [[7], [8]]
    assert [x.tolist() for x in events.before_window(loc, 2, offset=2)] == [[0], [1]]


def test_window_share_counts_the_topic_and_gives_nan_when_empty():
    feed = small_feed()
    cum = events.topic_cumsum(feed.topic, 3)
    loc = events.locate(feed, np.array([1, 2]), np.array([20, 5]))
    a, b = events.after_window(loc, 3)
    # user 1 after t=20: topics 9, 3, 3 -> 2/3 ; user 2: nothing after
    share = events.window_share(cum, a, b)
    assert share[0] == pytest.approx(2 / 3)
    assert np.isnan(share[1])
    a, b = events.before_window(loc, 100)
    assert events.window_share(cum, a, b)[0] == pytest.approx(2 / 3)


def test_end_within_and_left_within_use_minutes():
    df = pd.DataFrame({"user_id": [1, 1, 1], "time_ms": [0, 60_000, 600_000], "topic": [3, 3, 3], "tab": [1, 1, 1]})
    feed = events.build_feed(df)
    loc = events.locate(feed, np.array([1, 1]), np.array([0, 60_000]))
    # after t=0: a row at 1 minute and one at 10 minutes
    assert events.end_within(feed, loc, np.array([0, 60_000]), 5).tolist() == [2, 2]
    assert events.left_within(feed, loc, np.array([0, 60_000]), 5).tolist() == [False, True]
    assert events.left_within(feed, loc, np.array([0, 60_000]), 30).tolist() == [False, False]


def test_treated_matrix_and_share_matrix_shapes():
    feed = small_feed()
    treated = events.treated_matrix(np.array([3, 9, -1]), [3, 9])
    assert treated.tolist() == [[True, False, False], [False, True, False]]
    loc = events.locate(feed, np.array([1]), np.array([20]))
    a, b = events.before_window(loc, 100)
    shares = events.share_matrix(feed, [3, 9], a, b)
    assert shares[:, 0].tolist() == pytest.approx([2 / 3, 1 / 3])
