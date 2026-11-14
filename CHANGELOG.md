# Changelog

Process notes for this project: what I did in what order, problems found and re-runs.
The README holds only the current findings. Changes from the pre-registration go in
`DEVIATIONS.md`.

## 2026-11-14

Set up the repo and the data.

- Read the dataset pages before downloading. KuaiRand has three versions. I use
  KuaiRand-1K: 1,000 users drawn uniformly from the full 27,285, with all their logs.
  KuaiRand-Pure is smaller but keeps only logs on the 7,583 pool videos, so it does
  not contain the ordinary feed I need.
- Inclusion rule to remember: the full dataset keeps only users who received at least
  10 random videos in the two weeks. So every user here is a fairly active one.
- `scripts/download.py` fetches the archive from Zenodo, checks it against the
  published MD5 and extracts five files. It skips the 3.1 GB video statistics file,
  which I do not need. It also fetches the authors' supplementary category file, which
  gives names to topic IDs.
- `scripts/sanity.py` reproduces the six KuaiRand-1K counts in the paper's Table 1.
  All six match exactly (`results/sanity.md`).
- One difference from the paper worth knowing: random videos are 0.64% of impressions
  in the random period for these 1,000 users. The dataset page says about 0.37% for
  the full dataset. The page does not say how its figure was computed, so I only note it.
- Timestamps are per feed request. About six impressions share each timestamp, and a
  random video shares its timestamp with about five ordinary ones. So "the next 10
  impressions" has to start at the next request. The pre-registration defines this.
- The supplementary category file gives one named top-level category per video. It
  agrees with the main dataset's `tag` column for about 74% of videos, so the two are
  different labellings. I pre-registered the category file as the topic and the `tag`
  column as a check.
- `scripts/design_facts.py` writes the counts the pre-registration relies on. It reads
  the random log and the ordinary feed before 22 April only.
- Wrote `PREREGISTRATION.md` before writing any code that builds outcomes.
- Built the treated and control sets and the balance table (`results/balance.md`).
  Day, hour, activity and account age are balanced to within 0.003 SD. The baseline
  share of the topic is 0.020 SD lower for treated exposures (5.30% against 5.45%),
  and the interval does not include zero. That is inside the pre-registered limit of
  0.05 SD, so I continue. But it is a real, small tilt: people who already see a lot
  of a topic were slightly less likely to draw a random video in it. My guess is that
  the app does not re-show a video the user has already seen, and fans of a topic have
  seen more of its pool videos. I cannot check that guess from the logs. It matters
  for reading the results: the plain difference in outcomes starts about 0.15 points
  below zero, which is why the pre-registered estimator subtracts the baseline.
- Ran the event study (`results/event_study.md`). The pooled effect is -0.17 points in
  the next 10 impressions, -0.04 in the next 30 and +0.03 in the next 100. All three
  decision intervals are inside the half-point margin, so the pre-registered rule says
  noise. The next-10 interval is entirely below zero, which I report as its own small
  finding. Dropout is the same in both groups.
- Checked my guess about the tilt in the balance table. It holds as far as the logs
  can show: the same random video went to the same user twice only 2 times where a
  history-blind draw would give about 353 (`results/draw_check.md`). Recorded in
  `DEVIATIONS.md` as an addition.
- The pre-registered figure uses plain shares, so its "before" bins show the tilt and
  sit at about -0.15 instead of zero. I added a dashed line at that level.
- Ran the pre-registered splits (`results/heterogeneity.md`). New, long-registered,
  heavy and light users all look like the whole sample. Popular topics carry most of
  the next-10 dip. The one split that differs is watched against skipped: +0.33 points
  in the next 30 impressions after a watched random video, -0.12 after a skipped one.
  Only 17.4% of random videos were watched, so the average sits near zero.
- Added before-and-after bins for watched and skipped (see `DEVIATIONS.md`). Users who
  watched the random video were already 1.27 points above controls before it, so most
  of the watched-skipped gap is taste that was there already. The part that changes at
  the exposure is about +0.3 points for watched and -0.1 for skipped, and it fades.
- The two sources disagree on the data licence (CC BY-SA 4.0 on GitHub, CC BY 4.0 on
  Zenodo). It does not matter here because no data is committed.
