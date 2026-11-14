"""A small synthetic study shared by the script tests. No real data is needed."""
import numpy as np
import pandas as pd
import pytest

from src import events


def make_study(effect: float = 0.0, n_users: int = 120, seed: int = 0, targets=(3, 9)) -> events.Study:
    """Each user sees requests of 5 ordinary impressions. Each user has a taste for
    topic 3 and topic 9. Random exposures land in some requests with a random topic.
    With `effect` > 0, the 20 impressions after an exposure to topic T are more
    likely to be T."""
    rng = np.random.default_rng(seed)
    rows, evs = [], []
    for u in range(n_users):
        taste = {3: rng.uniform(0.05, 0.4), 9: rng.uniform(0.05, 0.3)}
        n_req = int(rng.integers(40, 80))
        boost = {3: 0, 9: 0}
        for r in range(n_req):
            t = 1_650_600_000_000 + r * 120_000 + u
            for _ in range(5):
                p3 = min(0.9, taste[3] + (effect if boost[3] > 0 else 0))
                p9 = min(0.9 - p3, taste[9] + (effect if boost[9] > 0 else 0))
                x = rng.random()
                topic = 3 if x < p3 else 9 if x < p3 + p9 else 12
                rows.append((u, t, topic, 1))
                for k in boost:
                    boost[k] = max(0, boost[k] - 1)
            if rng.random() < 0.25:
                ev_topic = int(rng.choice([3, 9, 12, -1]))
                evs.append((u, t, ev_topic, int(rng.random() < 0.3), int(rng.random() < 0.1)))
                if ev_topic in boost:
                    boost[ev_topic] = 20
    feed = events.build_feed(pd.DataFrame(rows, columns=["user_id", "time_ms", "topic", "tab"]))
    ev = pd.DataFrame(evs, columns=["user_id", "time_ms", "topic", "is_click", "long_view"])
    ev["date"] = 20220422 + (ev["time_ms"] % 3)
    ev["hourmin"] = 100 * (ev["time_ms"] % 24) + 30
    ev["tab"] = 1
    ev = pd.concat([ev, events.locate(feed, ev["user_id"].to_numpy(), ev["time_ms"].to_numpy())], axis=1)
    users = pd.DataFrame({"user_id": np.arange(n_users),
                          "register_days": rng.integers(20, 2000, n_users),
                          "pre_period_impressions": rng.integers(0, 9000, n_users)})
    return events.Study(feed, ev, list(targets), users)


@pytest.fixture(scope="session")
def null_study():
    return make_study(0.0)


@pytest.fixture(scope="session")
def effect_study():
    return make_study(0.15, seed=1)
