"""Exploratory splits of the main result, as pre-registered in section 10.

Same estimator inside each subgroup. No correction for the number of subgroups.
The watched-against-skipped split is description only: watching is the user's choice.
Writes results/heterogeneity.md, results/heterogeneity.json and results/heterogeneity.png.

    python scripts/heterogeneity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import estimate, events, outcomes  # noqa: E402

NEW_MAX_DAYS = 365
POPULAR = (39, 9, 12, 3, 20, 2)   # the six largest target topics by feed share before the experiment
INK, MUTED, GRID, SERIES = "#1f2328", "#59636e", "#d9dee3", "#2f6fb5"


def subgroups(study: events.Study) -> list[dict]:
    """Each subgroup as a mask over (topics, events), with a family and a label."""
    ev, users = study.events, study.users.set_index("user_id")
    n_topics, n_events = len(study.targets), len(ev)
    reg = ev["user_id"].map(users["register_days"]).to_numpy()
    pre = ev["user_id"].map(users["pre_period_impressions"]).to_numpy()
    lo_cut, hi_cut = users["pre_period_impressions"].quantile([1 / 3, 2 / 3])
    popular = np.isin(study.targets, POPULAR)

    def by_event(mask):
        return np.broadcast_to(np.asarray(mask, dtype=bool), (n_topics, n_events))

    def by_topic(mask):
        return np.broadcast_to(np.asarray(mask, dtype=bool)[:, None], (n_topics, n_events))

    def n_users(mask):
        return int(ev.loc[np.asarray(mask, dtype=bool), "user_id"].nunique())

    everyone = np.ones(n_events, dtype=bool)
    groups = [
        ("All", "All exposures", by_event(everyone), n_users(everyone)),
        ("Users", f"New: registered {NEW_MAX_DAYS} days or fewer", by_event(reg <= NEW_MAX_DAYS), n_users(reg <= NEW_MAX_DAYS)),
        ("Users", "Registered longer", by_event(reg > NEW_MAX_DAYS), n_users(reg > NEW_MAX_DAYS)),
        ("Users", "Heavy: top third by impressions before", by_event(pre > hi_cut), n_users(pre > hi_cut)),
        ("Users", "Light: bottom third by impressions before", by_event(pre <= lo_cut), n_users(pre <= lo_cut)),
        ("Topics", "Popular topics (6)", by_topic(popular), n_users(everyone)),
        ("Topics", "Niche topics (7)", by_topic(~popular), n_users(everyone)),
        ("Watching (descriptive)", "Random video watched", by_event(ev["is_click"] == 1), n_users(ev["is_click"] == 1)),
        ("Watching (descriptive)", "Random video skipped", by_event(ev["is_click"] == 0), n_users(ev["is_click"] == 0)),
        ("Watching (descriptive)", "Random video got a long view", by_event(ev["long_view"] == 1), n_users(ev["long_view"] == 1)),
        ("Watching (descriptive)", "No long view", by_event(ev["long_view"] == 0), n_users(ev["long_view"] == 0)),
    ]
    return [{"family": f, "label": l, "keep": k, "users": u} for f, l, k, u in groups]


def run(study: events.Study, weights: np.ndarray) -> list[dict]:
    rows = []
    for g in subgroups(study):
        effects = outcomes.window_effects(study, weights, keep=g["keep"])
        row = {"family": g["family"], "label": g["label"], "users": g["users"]}
        for k, r in effects.items():
            c = r["change"]
            lo, hi = c.interval(0.95)
            row[f"n_{k}"] = int(c.n_treated.sum())
            row[f"effect_{k}"], row[f"lo_{k}"], row[f"hi_{k}"] = c.pooled, lo, hi
        rows.append(row)
    return rows


def watching_bins(study: events.Study, weights: np.ndarray) -> dict:
    """Event-study bins separately for watched and skipped random videos. The bins before
    the exposure show how different the two groups already were. Not pre-registered."""
    ev = study.events
    return {"watched": outcomes.event_bins(study, weights, keep=(ev["is_click"] == 1).to_numpy()),
            "skipped": outcomes.event_bins(study, weights, keep=(ev["is_click"] == 0).to_numpy())}


def bin_summary(bins: list[dict]) -> dict:
    """Average difference over the bins before, and over the first three bins after."""
    before = [b["effect"] for b in bins if b["last"] < 0]
    after = [b["effect"] for b in bins if 1 <= b["first"] <= 21]
    return {"before": float(np.mean(before)), "last_before": float(before[-1]),
            "after_30": float(np.mean(after)), "rise": float(np.mean(after) - np.mean(before))}


def watch_rates(study: events.Study) -> dict:
    """How often random videos were watched, by whether the user already saw a lot of the topic."""
    ev = study.events
    in_target = np.isin(ev["topic"].to_numpy(), study.targets)
    return {"watched": float(ev["is_click"].mean()), "long_view": float(ev["long_view"].mean()),
            "watched_in_target_topics": float(ev.loc[in_target, "is_click"].mean())}


def pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def to_markdown(rows: list[dict], rates: dict, bins: dict | None = None) -> str:
    lines = [
        "# Heterogeneity (exploratory)",
        "",
        "Generated by `scripts/heterogeneity.py`. Same estimator as the main result inside each",
        "subgroup: change in the topic's share, treated minus control, in percentage points,",
        "with 95% intervals from a bootstrap over users. Nothing here is corrected for the",
        "number of subgroups, so single intervals that miss zero are not findings.",
        "",
        "| Split | Group | Users | Treated exposures | Next 10 | 95% interval | Next 30 | 95% interval | Next 100 | 95% interval |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---|",
    ]
    for r in rows:
        cells = " | ".join(f"{pp(r[f'effect_{k}'])} | {100 * r[f'lo_{k}']:+.2f} to {100 * r[f'hi_{k}']:+.2f}"
                           for k in outcomes.WINDOWS)
        lines.append(f"| {r['family']} | {r['label']} | {r['users']:,} | {r['n_30']:,} | {cells} |")
    lines += [
        "",
        f"Random videos were watched (a valid play) {100 * rates['watched']:.1f}% of the time and got a long view "
        f"{100 * rates['long_view']:.1f}% of the time.",
        "",
        "Watching is the user's choice. A user who watches a random video in a topic probably",
        "likes that topic already. The watched and skipped rows compare treated and control",
        "exposures with the same watching status, and they describe what followed. They do not",
        "show what watching causes.",
        "",
        "![Heterogeneity](heterogeneity.png)",
        "",
    ]
    if bins:
        lines += [
            "## Watched against skipped, before and after (not pre-registered)",
            "",
            "Plain treated-minus-control share of the topic in bins of 10 impressions, as in the",
            "main event study, split by whether the random video was watched. The level before the",
            "exposure shows how much the two groups differed already. See `DEVIATIONS.md`.",
            "",
            "![Watched against skipped](watched_skipped.png)",
            "",
            "| Random video | Average of the 10 bins before | Last bin before | Average of the 3 bins after | After minus before |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in ("watched", "skipped"):
            b = bin_summary(bins[name])
            lines.append(f"| {name.capitalize()} | {pp(b['before'])} | {pp(b['last_before'])} | "
                         f"{pp(b['after_30'])} | {pp(b['rise'])} |")
        lines.append("")
    return "\n".join(lines)


def watching_figure(bins: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"watched": "#c8501e", "skipped": SERIES}
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    for name, rows in bins.items():
        rows = [b for b in rows if b["first"] != 0]
        x = np.array([(b["first"] + b["last"]) / 2 for b in rows])
        y = np.array([100 * b["effect"] for b in rows])
        lo = np.array([100 * b["lo"] for b in rows])
        hi = np.array([100 * b["hi"] for b in rows])
        c = colours[name]
        ax.vlines(x + (0.8 if name == "watched" else -0.8), lo, hi, color=c, lw=1.2, alpha=0.45)
        for side in (x < 0, x > 0):
            ax.plot(x[side], y[side], color=c, lw=2, marker="o", ms=4)
        ax.annotate(f"random video {name}", xy=(x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    ax.set_xlim(-104, 150)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["100 before", "50 before", "random\nvideo", "50 after", "100 after"])
    ax.set_ylabel("Treated minus control share of the topic\n(percentage points)", color=INK, fontsize=9)
    ax.set_xlabel("Ordinary impressions before and after the random video, in bins of 10", color=INK, fontsize=9)
    ax.set_title("Watched and skipped random videos, before and after", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), dpi=200, sharey=True, sharex=True)
    for ax, k in zip(axes, outcomes.WINDOWS):
        eff = np.array([100 * r[f"effect_{k}"] for r in rows])
        lo = np.array([100 * r[f"lo_{k}"] for r in rows])
        hi = np.array([100 * r[f"hi_{k}"] for r in rows])
        ax.axvline(0, color=MUTED, lw=1)
        ax.hlines(y, lo, hi, color=SERIES, lw=1.5, alpha=0.55)
        ax.plot(eff, y, "o", color=SERIES, ms=5)
        ax.set_title(f"Next {k} impressions", loc="left", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, color=INK, fontsize=8)
    axes[1].set_xlabel("Change in the topic's share, treated minus control (percentage points)", color=INK, fontsize=9)
    fig.suptitle("Effect of one random video on its topic's share of the feed, by subgroup", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    study = events.load_study(args.raw_dir, args.processed_dir)
    weights = estimate.bootstrap_weights(study.events["user_id"].nunique())
    rows = run(study, weights)
    rates = watch_rates(study)
    bins = watching_bins(study, weights)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: bin_summary(v) for k, v in bins.items()}
    (out / "heterogeneity.json").write_text(json.dumps(
        {"rows": rows, "rates": rates, "watching_bins": bins, "watching_summary": summary}, indent=2), encoding="utf-8")
    md = to_markdown(rows, rates, bins)
    (out / "heterogeneity.md").write_text(md, encoding="utf-8")
    figure(rows, out / "heterogeneity.png")
    watching_figure(bins, out / "watched_skipped.png")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
