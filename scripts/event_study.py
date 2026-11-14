"""Main result: does the feed show more of a topic after one random video in it?

Writes results/event_study.md, results/event_study.json and results/event_study.png.

    python scripts/event_study.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import estimate, events, outcomes, topics  # noqa: E402

INK, MUTED, GRID, SERIES = "#1f2328", "#59636e", "#d9dee3", "#2f6fb5"


def pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def span(lo: float, hi: float) -> str:
    return f"{100 * lo:+.2f} to {100 * hi:+.2f}"


def summarise(effects: dict) -> list[dict]:
    """One row per window from outcomes.window_effects."""
    rows = []
    for k, r in effects.items():
        c, p = r["change"], r["plain"]
        lo, hi = c.interval(0.95)
        dlo, dhi = c.interval(outcomes.DECISION_LEVEL)
        plo, phi = p.interval(0.95)
        rows.append({"window": k, "n_treated": int(c.n_treated.sum()),
                     "treated_change": c.pooled_treated_mean, "control_change": c.pooled_control_mean,
                     "effect": c.pooled, "lo": lo, "hi": hi, "decision_lo": dlo, "decision_hi": dhi,
                     "treated_share": p.pooled_treated_mean, "control_share": p.pooled_control_mean,
                     "plain": p.pooled, "plain_lo": plo, "plain_hi": phi})
    return rows


def by_topic(effects: dict, targets: list[int]) -> list[dict]:
    rows = []
    for i, t in enumerate(targets):
        row = {"topic": t, "name": topics.ENGLISH.get(t, str(t))}
        for k, r in effects.items():
            c = r["change"]
            ci = c.topic_intervals(0.95)
            row[f"n_{k}"] = int(c.n_treated[i])
            row[f"effect_{k}"] = float(c.effect[i])
            row[f"lo_{k}"], row[f"hi_{k}"] = float(ci[i, 0]), float(ci[i, 1])
        rows.append(row)
    return rows


def dropout_rows(d: dict) -> list[dict]:
    rows = []
    for key, label in (("left", f"No later impression within {outcomes.LEFT_MINUTES} minutes"),
                       ("none_after", "No later impression before the log ends")):
        r = d[key]
        lo, hi = r.interval(0.95)
        rows.append({"measure": label, "treated": r.pooled_treated_mean, "control": r.pooled_control_mean,
                     "difference": r.pooled, "lo": lo, "hi": hi})
    return rows


def figure(bins: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([0 if b["first"] == 0 else (b["first"] + b["last"]) / 2 for b in bins])
    y = np.array([100 * b["effect"] for b in bins])
    lo = np.array([100 * b["lo"] for b in bins])
    hi = np.array([100 * b["hi"] for b in bins])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax.axhline(0, color=MUTED, lw=1)
    before = float(np.mean(y[x < 0]))
    ax.axhline(before, color=SERIES, lw=1, ls=(0, (4, 3)), alpha=0.7)
    ax.annotate("average before", xy=(102, before), xytext=(0, -11), textcoords="offset points",
                color=MUTED, fontsize=8, ha="right",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.vlines(x, lo, hi, color=SERIES, lw=1.5, alpha=0.55)
    ax.plot(x[x < 0], y[x < 0], color=SERIES, lw=2, marker="o", ms=5)
    ax.plot(x[x > 0], y[x > 0], color=SERIES, lw=2, marker="o", ms=5)
    ax.plot(x[x == 0], y[x == 0], color=SERIES, marker="o", ms=5, mfc="white", mew=1.5, ls="none")
    limit = max(0.6, 1.15 * max(abs(lo).max(), abs(hi).max()))
    ax.set_ylim(-limit, limit)
    ax.set_xlim(-104, 104)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["100 before", "50 before", "random\nvideo", "50 after", "100 after"])
    ax.annotate("same feed request", xy=(0, y[x == 0][0]), xytext=(8, limit * 0.72), color=MUTED, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylabel("Treated minus control share of the topic\n(percentage points)", color=INK, fontsize=9)
    ax.set_xlabel("Ordinary impressions before and after the random video, in bins of 10", color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


TITLES = {
    "the feed bends": "One random video in a topic raises that topic's share of the feed",
    "noise": "One random video in a topic does not raise that topic's share of the feed",
    "the feed moves away": "After one random video in a topic, the feed shows less of that topic",
    "undecided": "After one random video in a topic: too noisy to call",
}


def to_markdown(res: dict) -> str:
    m = res["main"]
    lines = [
        "# Event study: the feed after one random video",
        "",
        "Generated by `scripts/event_study.py`. Effects are in percentage points of feed share,",
        "pooled over the 13 target topics. Intervals come from 2,000 bootstrap resamples of users.",
        "",
        "## Main result (pre-registered)",
        "",
        "Change is the topic's share in the window minus its share in the 100 impressions before.",
        "",
        "| Window | Treated exposures | Treated change | Control change | Effect | 95% interval | 98.33% interval |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in m:
        lines.append(f"| Next {r['window']} | {r['n_treated']:,} | {pp(r['treated_change'])} | {pp(r['control_change'])} | "
                     f"{pp(r['effect'])} | {span(r['lo'], r['hi'])} | {span(r['decision_lo'], r['decision_hi'])} |")
    lines += [
        "",
        f"Decision by the pre-registered rule (98.33% intervals, margin ±{100 * outcomes.MARGIN:.1f} points): "
        f"**{res['decision']}**.",
        "",
    ]
    below = [r["window"] for r in m if r["decision_hi"] < 0]
    if below:
        lines += [
            "The pre-registration also says an interval entirely below zero is reported as its own",
            "finding. That holds for the next " + ", ".join(str(k) for k in below) + " impressions: "
            "the feed shows slightly less of the topic there.",
            "",
        ]
    lines += [
        f"Exposures used: {res['counts']['with_outcome']:,} of {res['counts']['events']:,}. "
        f"{res['counts']['no_before']:,} have no impression before and {res['counts']['no_after']:,} have none after.",
        "",
        "## Plain difference, no baseline subtracted",
        "",
        "| Window | Treated share | Control share | Difference | 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for r in m:
        lines.append(f"| Next {r['window']} | {100 * r['treated_share']:.2f}% | {100 * r['control_share']:.2f}% | "
                     f"{pp(r['plain'])} | {span(r['plain_lo'], r['plain_hi'])} |")
    lines += [
        "",
        "## Event study in bins of 10 impressions",
        "",
        "![Event study](event_study.png)",
        "",
        "Plain shares, treated minus control. Bins before the exposure are a placebo. They sit",
        f"at {pp(res['before_mean'])} on average, the same tilt the balance table shows. The dashed line marks it.",
        "",
        "| Impressions | Treated share | Control share | Difference | 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for b in res["bins"]:
        label = "same request" if b["first"] == 0 else f"{b['first']:+d} to {b['last']:+d}"
        lines.append(f"| {label} | {100 * b['treated_mean']:.2f}% | {100 * b['control_mean']:.2f}% | "
                     f"{pp(b['effect'])} | {span(b['lo'], b['hi'])} |")
    lines += [
        "",
        "## Dropout",
        "",
        "| Measure | Treated | Control | Difference | 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for r in res["dropout"]:
        lines.append(f"| {r['measure']} | {100 * r['treated']:.2f}% | {100 * r['control']:.2f}% | "
                     f"{pp(r['difference'])} | {span(r['lo'], r['hi'])} |")
    lines += [
        "",
        "## Checks (pre-registered)",
        "",
        "Same estimator as the main result, with one thing changed.",
        "",
        "| Check | Window | Treated exposures | Effect | 95% interval |",
        "|---|---|---:|---:|---|",
    ]
    for name, rows in res["checks"].items():
        for r in rows:
            lines.append(f"| {name} | Next {r['window']} | {r['n_treated']:,} | {pp(r['effect'])} | {span(r['lo'], r['hi'])} |")
    lines += [
        "",
        "## By topic (exploratory, no correction for 13 topics)",
        "",
        "| Topic | Treated | Next 10 | 95% interval | Next 30 | 95% interval | Next 100 | 95% interval |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for r in res["by_topic"]:
        lines.append(f"| {r['name']} | {r['n_30']:,} | " + " | ".join(
            f"{pp(r[f'effect_{k}'])} | {span(r[f'lo_{k}'], r[f'hi_{k}'])}" for k in outcomes.WINDOWS) + " |")
    lines.append("")
    return "\n".join(lines)


def run(study: events.Study, weights: np.ndarray) -> dict:
    """Everything that needs only the main study."""
    effects = outcomes.window_effects(study, weights)
    main = summarise(effects)
    ev = study.events
    bins = outcomes.event_bins(study, weights)
    return {
        "before_mean": float(np.mean([b["effect"] for b in bins if b["last"] < 0])),
        "main": main,
        "decision": outcomes.decide([(r["decision_lo"], r["decision_hi"]) for r in main]),
        "by_topic": by_topic(effects, study.targets),
        "bins": bins,
        "dropout": dropout_rows(outcomes.dropout(study, weights)),
        "counts": {"events": int(len(ev)), "no_before": int((ev["lo"] == ev["s"]).sum()),
                   "no_after": int((ev["hi"] == ev["e"]).sum()),
                   "with_outcome": int(((ev["lo"] > ev["s"]) & (ev["hi"] < ev["e"])).sum())},
        "checks": {f"Only impressions within {outcomes.NEAR_MINUTES} minutes": summarise(
            outcomes.window_effects(study, weights, end=outcomes.near_end(study)))},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    study = events.load_study(args.raw_dir, args.processed_dir)
    n_users = study.events["user_id"].nunique()
    weights = estimate.bootstrap_weights(n_users)
    res = run(study, weights)
    del study
    tab1 = events.load_study(args.raw_dir, args.processed_dir, tab=1)
    res["checks"]["Feed limited to tab 1"] = summarise(outcomes.window_effects(tab1, weights))
    del tab1
    tags = events.load_study(args.raw_dir, args.processed_dir, use_tags=True)
    res["checks"]["Topics from the tag column"] = summarise(outcomes.window_effects(tags, weights))
    del tags

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "event_study.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "event_study.md").write_text(to_markdown(res), encoding="utf-8")
    figure(res["bins"], out / "event_study.png", TITLES[res["decision"]])
    sys.stdout.buffer.write(to_markdown(res).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
