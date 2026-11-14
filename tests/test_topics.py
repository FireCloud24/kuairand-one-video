"""Topic lookup, target choice and the block-noise measure."""
import numpy as np
import pandas as pd
import pytest

from scripts import design_facts
from src import topics


def test_clean_topic_maps_unknown_and_missing_to_no_topic():
    out = topics.clean_topic(pd.Series([39.0, -124.0, None, 3.0]))
    assert out.tolist() == [39, topics.NO_TOPIC, topics.NO_TOPIC, 3]


def test_build_video_topics_filters_in_chunks(tmp_path):
    path = tmp_path / "cats.csv"
    pd.DataFrame({
        "final_video_id": [0, 1, 2, 3, 4, 5],
        "first_level_category_id": [39.0, 2.0, -124.0, 3.0, None, 12.0],
        "first_level_category_name": ["a"] * 6,
    }).to_csv(path, index=False)
    out = topics.build_video_topics(path, [5, 1, 2, 4, 99], chunksize=2)
    assert dict(zip(out.video_id, out.topic)) == {1: 2, 2: topics.NO_TOPIC, 4: topics.NO_TOPIC, 5: 12}


def test_attach_topic_gives_no_topic_to_unknown_videos():
    log = pd.DataFrame({"video_id": [10, 11, 12]})
    video_topics = pd.DataFrame({"video_id": [10, 12], "topic": [3, 39]})
    assert topics.attach_topic(log, video_topics).tolist() == [3, topics.NO_TOPIC, 39]


def test_topic_counts_and_choose_targets():
    topic = pd.Series([3, 3, 3, 39, 39, topics.NO_TOPIC, topics.NO_TOPIC, topics.NO_TOPIC, topics.NO_TOPIC])
    users = pd.Series([1, 1, 2, 1, 3, 1, 2, 3, 4])
    videos = pd.Series([7, 7, 8, 9, 9, 1, 2, 3, 4])
    counts = topics.topic_counts(topic, users, videos)
    assert counts.loc[3].tolist() == [3, 2, 2]
    assert counts.loc[39].tolist() == [2, 2, 1]
    # NO_TOPIC has the most exposures but is never a target
    assert topics.choose_targets(counts, 2) == [3, 39]
    assert topics.choose_targets(counts, 3) == [3]


def test_block_share_sd_uses_full_blocks_within_each_user():
    # user 1: blocks [1,1] and [0,0] (last row dropped); user 2: block [1,0]
    is_topic = np.array([1, 1, 0, 0, 1, 1, 0])
    user = np.array([1, 1, 1, 1, 1, 2, 2])
    sd = topics.block_share_sd(is_topic, user, 2)
    assert sd == pytest.approx(np.std([1.0, 0.0, 0.5], ddof=1))


def test_block_share_sd_is_nan_without_a_full_block():
    assert np.isnan(topics.block_share_sd(np.array([1, 0]), np.array([1, 2]), 2))


def test_minimum_detectable_shrinks_with_more_data():
    small = design_facts.minimum_detectable(0.1, 100, 1000)
    large = design_facts.minimum_detectable(0.1, 10000, 100000)
    assert small == pytest.approx(2.8 * 0.1 * np.sqrt(1 / 100 + 1 / 1000))
    assert large < small
