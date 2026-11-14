"""Random exposures as events, placed in each user's ordinary feed.

The ordinary feed is one long array sorted by user, then timestamp, then file order.
Each random exposure gets four positions in that array:

    s   start of the user's impressions
    lo  first impression with the same timestamp as the exposure (same feed request)
    hi  first impression with a later timestamp
    e   end of the user's impressions

So [s, lo) is "before", [lo, hi) is the same request, and [hi, e) is "after".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, topics

MIN_EXPOSURES = 1000   # pre-registered threshold for a target topic
BASELINE_N = 100       # pre-registered: last 100 ordinary impressions before the exposure


@dataclass
class Feed:
    """Ordinary impressions sorted by (user, time, file order)."""
    user: np.ndarray
    time: np.ndarray
    topic: np.ndarray
    tab: np.ndarray
    order: np.ndarray | None = None   # positions of these rows in the frame they came from

    def __len__(self) -> int:
        return len(self.user)


def build_feed(standard: pd.DataFrame) -> Feed:
    """standard needs user_id, time_ms, topic, tab. A stable sort keeps file order
    among rows that share a timestamp."""
    order = np.lexsort((np.arange(len(standard)), standard["time_ms"].to_numpy(), standard["user_id"].to_numpy()))
    s = standard.iloc[order]
    return Feed(s["user_id"].to_numpy(), s["time_ms"].to_numpy(), s["topic"].to_numpy(), s["tab"].to_numpy(), order)


def locate(feed: Feed, ev_user: np.ndarray, ev_time: np.ndarray) -> pd.DataFrame:
    """Positions s, lo, hi, e for each event. Users with no impressions get s = e."""
    ev_user = np.asarray(ev_user)
    ev_time = np.asarray(ev_time)
    s = np.searchsorted(feed.user, ev_user, side="left")
    e = np.searchsorted(feed.user, ev_user, side="right")
    lo = np.empty(len(ev_user), dtype=np.int64)
    hi = np.empty(len(ev_user), dtype=np.int64)
    for i in range(len(ev_user)):
        times = feed.time[s[i]:e[i]]
        lo[i] = s[i] + np.searchsorted(times, ev_time[i], side="left")
        hi[i] = s[i] + np.searchsorted(times, ev_time[i], side="right")
    return pd.DataFrame({"s": s, "lo": lo, "hi": hi, "e": e})


def end_within(feed: Feed, loc: pd.DataFrame, ev_time: np.ndarray, minutes: float) -> np.ndarray:
    """For each event, the position just past the last impression that is no more than
    `minutes` after the exposure. Always between hi and e."""
    limit = np.asarray(ev_time) + int(minutes * 60_000)
    out = np.empty(len(loc), dtype=np.int64)
    hi, e = loc["hi"].to_numpy(), loc["e"].to_numpy()
    for i in range(len(loc)):
        out[i] = hi[i] + np.searchsorted(feed.time[hi[i]:e[i]], limit[i], side="right")
    return out


def topic_cumsum(feed_topic: np.ndarray, topic: int) -> np.ndarray:
    """c with c[i] = number of the first i impressions that are in `topic`."""
    return np.concatenate([[0], np.cumsum(feed_topic == topic, dtype=np.int64)])


def window_share(cum: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of impressions in [a, b) that are in the topic; NaN for an empty window."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = (b - a).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cum[b] - cum[a]) / n, np.nan)


def before_window(loc: pd.DataFrame, n: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The n impressions ending `offset` impressions before the same-request block,
    cut at the start of the user's feed."""
    s, lo = loc["s"].to_numpy(), loc["lo"].to_numpy()
    b = np.maximum(s, lo - offset)
    return np.maximum(s, b - n), b


def after_window(loc: pd.DataFrame, k: int, offset: int = 0, end: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The k impressions starting `offset` impressions after the same-request block,
    cut at the end of the user's feed (or at `end` when given)."""
    hi = loc["hi"].to_numpy()
    e = loc["e"].to_numpy() if end is None else np.asarray(end)
    a = np.minimum(e, hi + offset)
    return a, np.minimum(e, a + k)


def left_within(feed: Feed, loc: pd.DataFrame, ev_time: np.ndarray, minutes: float) -> np.ndarray:
    """True when no ordinary impression from a later request arrives within `minutes`."""
    return end_within(feed, loc, ev_time, minutes) == loc["hi"].to_numpy()


def treated_matrix(event_topic: np.ndarray, targets: list[int]) -> np.ndarray:
    """(topics, events) boolean: is the event's random video in that topic?"""
    return np.asarray(event_topic)[None, :] == np.asarray(targets)[:, None]


def share_matrix(feed: Feed, targets: list[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(topics, events): share of each target topic in the window [a, b) of each event."""
    return np.vstack([window_share(topic_cumsum(feed.topic, t), a, b) for t in targets])


@dataclass
class Study:
    feed: Feed
    events: pd.DataFrame     # one row per random exposure, with s, lo, hi, e
    targets: list[int]
    users: pd.DataFrame      # user features plus pre-period activity
    cums: list | None = None           # one topic_cumsum per target; built on first use
    treated: np.ndarray | None = None  # (topics, events); built on first use

    def prepare(self) -> "Study":
        if self.cums is None:
            self.cums = [topic_cumsum(self.feed.topic, t) for t in self.targets]
        if self.treated is None:
            self.treated = treated_matrix(self.events["topic"].to_numpy(), self.targets)
        return self

    def shares(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(topics, events): share of each target topic in the window [a, b)."""
        self.prepare()
        return np.vstack([window_share(c, a, b) for c in self.cums])


PRE_PERIOD_END = 20220421


def tag_membership(video_ids: np.ndarray, video_tags: pd.DataFrame, targets: list[int]) -> np.ndarray:
    """(topics, rows) boolean: does the row's video list that id in its `tag` string?
    video_tags has video_id and tag, where tag looks like "39" or "12,62" or is missing."""
    tags = video_tags.dropna(subset=["tag"])
    pairs = tags.assign(tag=tags["tag"].astype(str).str.split(",")).explode("tag")
    pairs["tag"] = pd.to_numeric(pairs["tag"], errors="coerce")
    out = np.zeros((len(targets), len(video_ids)), dtype=bool)
    for i, t in enumerate(targets):
        out[i] = np.isin(video_ids, pairs.loc[pairs["tag"] == t, "video_id"].to_numpy())
    return out


def load_study(raw_dir: Path | str, processed_dir: Path | str, tab: int | None = None,
               use_tags: bool = False) -> Study:
    """Load everything and place every random exposure in the feed.
    With `tab`, only ordinary impressions in that tab make up the feed.
    With `use_tags`, topic membership comes from the main dataset's `tag` column."""
    video_topics = topics.load_video_topics(raw_dir, processed_dir)
    random = data.load_random(raw_dir, ["user_id", "video_id", "date", "hourmin", "time_ms",
                                        "is_click", "long_view", "tab"])
    random["topic"] = topics.attach_topic(random, video_topics)
    counts = topics.topic_counts(random["topic"], random["user_id"], random["video_id"])
    targets = topics.choose_targets(counts, MIN_EXPOSURES)

    standard = data.load_standard(raw_dir, ["user_id", "video_id", "date", "time_ms", "tab"])
    pre_counts = standard[standard["date"] <= PRE_PERIOD_END].groupby("user_id").size()
    standard["topic"] = topics.attach_topic(standard, video_topics)
    if tab is not None:
        standard = standard[standard["tab"] == tab]
    feed = build_feed(standard)

    events = random.reset_index(drop=True)
    events = pd.concat([events, locate(feed, events["user_id"].to_numpy(), events["time_ms"].to_numpy())], axis=1)
    users = pd.read_csv(Path(raw_dir) / data.USER_FEATURES, usecols=["user_id", "register_days"])
    users["pre_period_impressions"] = users["user_id"].map(pre_counts).fillna(0).astype(int)
    study = Study(feed, events, targets, users)
    if use_tags:
        video_tags = pd.read_csv(Path(raw_dir) / data.VIDEO_FEATURES, usecols=["video_id", "tag"],
                                 dtype={"tag": "string"})
        member = tag_membership(standard["video_id"].to_numpy()[feed.order], video_tags, targets)
        study.cums = [np.concatenate([[0], np.cumsum(row, dtype=np.int64)]) for row in member]
        study.treated = tag_membership(events["video_id"].to_numpy(), video_tags, targets)
    return study
