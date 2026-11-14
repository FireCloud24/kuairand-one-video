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
- The two sources disagree on the data licence (CC BY-SA 4.0 on GitHub, CC BY 4.0 on
  Zenodo). It does not matter here because no data is committed.
