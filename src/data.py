"""Load the KuaiRand-1K files and count what is in them."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RANDOM_LOG = "log_random_4_22_to_5_08_1k.csv"
STANDARD_LOGS = ("log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv")
USER_FEATURES = "user_features_1k.csv"
VIDEO_FEATURES = "video_features_basic_1k.csv"
CATEGORIES = "kuairand_video_categories.csv"

# The twelve feedback signals in each log row.
SIGNALS = (
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter",
)
BINARY_SIGNALS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
                  "long_view", "is_profile_enter")
# Columns the analysis needs. Small integer types keep 11.7 million rows in memory.
LOG_DTYPES = {
    "user_id": "int32", "video_id": "int64", "date": "int32", "hourmin": "int16",
    "time_ms": "int64", "is_click": "int8", "is_like": "int8", "is_follow": "int8",
    "is_comment": "int8", "is_forward": "int8", "is_hate": "int8", "long_view": "int8",
    "play_time_ms": "int64", "duration_ms": "int64", "profile_stay_time": "int64",
    "comment_stay_time": "int64", "is_profile_enter": "int8", "is_rand": "int8", "tab": "int8",
}


def load_log(path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one log file, keeping only `columns` (all known columns by default)."""
    columns = list(columns) if columns is not None else list(LOG_DTYPES)
    dtypes = {c: LOG_DTYPES[c] for c in columns}
    return pd.read_csv(path, usecols=columns, dtype=dtypes)[columns]


def load_random(raw_dir: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    return load_log(Path(raw_dir) / RANDOM_LOG, columns)


def load_standard(raw_dir: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Both ordinary-feed logs, 8 April to 8 May 2022, in file order."""
    parts = [load_log(Path(raw_dir) / name, columns) for name in STANDARD_LOGS]
    return pd.concat(parts, ignore_index=True)


def basic_stats(log: pd.DataFrame) -> dict:
    """Users, videos and rows in a log: the three counts in the paper's Table 1."""
    return {
        "users": int(log["user_id"].nunique()),
        "items": int(log["video_id"].nunique()),
        "interactions": int(len(log)),
    }


def signal_rates(log: pd.DataFrame) -> dict:
    """Share of rows where each yes/no feedback signal is 1."""
    return {s: float(log[s].mean()) for s in BINARY_SIGNALS if s in log.columns}


def per_user_counts(log: pd.DataFrame) -> pd.Series:
    """Number of rows per user, indexed by user_id."""
    return log.groupby("user_id").size()
