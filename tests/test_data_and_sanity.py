"""Loading logs and the counts behind the sanity table."""
import pandas as pd

from scripts import sanity
from src import data


def tiny_log():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 2],
        "video_id": [10, 11, 10, 12, 12],
        "date": [20220422, 20220423, 20220422, 20220424, 20220508],
        "is_rand": [1, 1, 1, 1, 1],
        "is_click": [1, 0, 0, 1, 1],
        "long_view": [0, 0, 0, 0, 1],
    })


def test_basic_stats_counts_distinct_users_and_videos():
    assert data.basic_stats(tiny_log()) == {"users": 2, "items": 3, "interactions": 5}


def test_signal_rates_are_row_shares():
    rates = data.signal_rates(tiny_log())
    assert rates["is_click"] == 0.6
    assert rates["long_view"] == 0.2
    assert "is_like" not in rates


def test_per_user_counts():
    counts = data.per_user_counts(tiny_log())
    assert counts.loc[1] == 2 and counts.loc[2] == 3


def test_load_log_keeps_requested_columns_and_types(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame({"user_id": [1], "video_id": [5], "date": [20220422], "hourmin": [400],
                  "time_ms": [1650485801301], "is_rand": [1], "tab": [1], "extra": [9]}).to_csv(path, index=False)
    log = data.load_log(path, ["user_id", "time_ms", "is_rand"])
    assert list(log.columns) == ["user_id", "time_ms", "is_rand"]
    assert log["time_ms"].iloc[0] == 1650485801301
    assert str(log["is_rand"].dtype) == "int8"


def test_compare_flags_mismatches():
    ours = {"standard": {"users": 1000, "items": 5, "interactions": 7},
            "random": {"users": 1000, "items": 7388, "interactions": 43028}}
    rows = sanity.compare(ours, sanity.PAPER)
    by = {(r["policy"], r["statistic"]): r for r in rows}
    assert by[("random", "interactions")]["match"] is True
    assert by[("standard", "items")]["match"] is False
    assert by[("standard", "items")]["difference"] == 5 - 4369953


def test_describe_and_markdown():
    random = tiny_log()
    standard = pd.DataFrame({"user_id": [1, 2, 2, 2], "video_id": [1, 2, 3, 4],
                             "date": [20220408, 20220422, 20220423, 20220508], "is_rand": [0, 0, 0, 0]})
    extra = sanity.describe(random, standard)
    assert extra["random_log_is_rand_share"] == 1.0
    assert extra["standard_log_is_rand_share"] == 0.0
    assert extra["random_per_user_min"] == 2
    # 5 random rows and 3 ordinary rows on or after 22 April
    assert extra["random_share_of_period"] == 5 / 8
    md = sanity.to_markdown(sanity.compare(
        {"standard": data.basic_stats(standard), "random": data.basic_stats(random)}, sanity.PAPER), extra)
    assert "| random | users | 1,000 | 2 | -998 | NO |" in md
