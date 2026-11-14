"""Video topics: one top-level category per video.

The topic of a video is its first-level category in the authors' supplementary
file `kuairand_video_categories.csv`. That file covers all 32 million videos of
the full dataset, so the first job is to cut it down to the videos in KuaiRand-1K.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data

NO_TOPIC = -1  # videos with no category row or an unknown category

# My translations of the first-level category names in the supplementary file.
ENGLISH = {
    1: "Dance", 2: "Music", 3: "Games", 4: "Beauty", 5: "Fashion", 6: "Celebrities",
    7: "Sports", 8: "Good looks", 9: "Comedy", 10: "Travel", 11: "Daily life", 12: "Food",
    13: "Farming and rural life", 14: "Education", 15: "Art", 16: "Health", 17: "Pets",
    18: "Cars", 19: "Relationships", 20: "Anime and comics", 21: "Humanities", 22: "Finance",
    23: "Current affairs", 24: "Astrology", 25: "Parenting", 26: "Photography",
    27: "Tech and gadgets", 28: "Local news", 29: "Science and law", 31: "Fitness",
    32: "Short drama", 33: "Selfies", 34: "Other", 35: "Military", 36: "Home and property",
    37: "Oddities", 38: "Books", 39: "Film and TV",
}


def clean_topic(ids: pd.Series) -> pd.Series:
    """Category ids as int16, with missing or non-positive values as NO_TOPIC."""
    ids = pd.to_numeric(ids, errors="coerce")
    return ids.where(ids > 0, NO_TOPIC).fillna(NO_TOPIC).astype("int16")


def build_video_topics(categories_csv: Path | str, video_ids, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Read the big category file in chunks and keep the rows for `video_ids`.
    Returns columns video_id, topic (NO_TOPIC when unknown), one row per video found."""
    wanted = pd.Index(np.unique(np.asarray(video_ids)))
    parts = []
    cols = ["final_video_id", "first_level_category_id"]
    for chunk in pd.read_csv(categories_csv, usecols=cols, chunksize=chunksize):
        keep = chunk[chunk["final_video_id"].isin(wanted)]
        parts.append(pd.DataFrame({"video_id": keep["final_video_id"].astype("int64").to_numpy(),
                                   "topic": clean_topic(keep["first_level_category_id"]).to_numpy()}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"video_id": [], "topic": []})
    return out.drop_duplicates("video_id").reset_index(drop=True)


def load_video_topics(raw_dir: Path | str, processed_dir: Path | str, video_ids=None) -> pd.DataFrame:
    """Cached version of build_video_topics for the videos in the 1K feature file."""
    cache = Path(processed_dir) / "video_topics.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"video_id": "int64", "topic": "int16"})
    if video_ids is None:
        video_ids = pd.read_csv(Path(raw_dir) / data.VIDEO_FEATURES, usecols=["video_id"])["video_id"]
    topics = build_video_topics(Path(raw_dir) / data.CATEGORIES, video_ids)
    cache.parent.mkdir(parents=True, exist_ok=True)
    topics.to_csv(cache, index=False)
    return topics


def attach_topic(log: pd.DataFrame, video_topics: pd.DataFrame) -> pd.Series:
    """Topic of each log row (NO_TOPIC when the video has none), aligned to log.index."""
    lookup = video_topics.set_index("video_id")["topic"]
    return log["video_id"].map(lookup).fillna(NO_TOPIC).astype("int16")


def topic_counts(random_topic: pd.Series, random_users: pd.Series, random_videos: pd.Series) -> pd.DataFrame:
    """Per topic: random exposures, distinct users exposed, distinct pool videos."""
    df = pd.DataFrame({"topic": random_topic.to_numpy(), "user": random_users.to_numpy(),
                       "video": random_videos.to_numpy()})
    out = df.groupby("topic").agg(random_exposures=("user", "size"), users=("user", "nunique"),
                                  pool_videos=("video", "nunique"))
    return out.sort_values("random_exposures", ascending=False)


def choose_targets(counts: pd.DataFrame, min_exposures: int) -> list[int]:
    """Topics with at least `min_exposures` random exposures, most exposed first.
    NO_TOPIC is never a target."""
    keep = counts[(counts["random_exposures"] >= min_exposures) & (counts.index != NO_TOPIC)]
    return [int(t) for t in keep.sort_values("random_exposures", ascending=False).index]


def block_share_sd(is_topic: np.ndarray, user: np.ndarray, k: int) -> float:
    """Cut each user's impressions (already in time order) into consecutive blocks of
    k and return the standard deviation of the topic's share across all full blocks.
    This says how noisy a k-impression window is before any experiment."""
    is_topic = np.asarray(is_topic, dtype=float)
    user = np.asarray(user)
    shares = []
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    ends = np.r_[starts[1:], len(user)]
    for s, e in zip(starts, ends):
        n = (e - s) // k
        if n:
            shares.append(is_topic[s:s + n * k].reshape(n, k).mean(axis=1))
    if not shares:
        return float("nan")
    return float(np.concatenate(shares).std(ddof=1))
