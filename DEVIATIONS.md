# Deviations from the pre-registration

Every change from `PREREGISTRATION.md`, with the date and the reason. Nothing in the
pre-registered analysis was dropped or redefined. The entries below are additions.

## 2026-11-14: a check on how the random video was drawn (added)

The balance table showed treated exposures starting with a slightly lower share of the
topic (0.020 SD, inside the pre-registered limit of 0.05). I added
`scripts/draw_check.py` to find out why. It counts how often a user was given the same
random video twice, and how often the random video had already been in that user's
ordinary feed. Both are far rarer than a draw that ignored history would produce
(`results/draw_check.md`). So the random video was drawn from videos the user had not
been shown yet. A fan of a topic has seen more of its pool videos, so is slightly less
likely to draw that topic. This check does not use any outcome.

Effect on the analysis: none on the pre-registered estimator, which subtracts each
exposure's own baseline. It does change how to read the plain difference and the
event-study figure. Both start about 0.15 points below zero before the exposure.

## 2026-11-14: a dashed line on the event-study figure (added)

The pre-registered figure uses plain shares so that the bins before the exposure act
as a placebo. Because of the tilt above, those bins sit below zero. I added a dashed
line at their average so the reader compares the bins after the exposure with the
right level. The points and intervals are as pre-registered.

## 2026-11-14: before-and-after bins for watched and skipped (added)

The pre-registered watched-against-skipped split gave opposite signs: the topic's share
rose after a watched random video and fell after a skipped one. Before describing
that, I wanted to see how different the two groups already were. So I added the
event-study bins split by watching status (`results/heterogeneity.md`, second section,
and `results/watched_skipped.png`). I added it after seeing the split, so it is
exploratory twice over. It supports the warning in the pre-registration: before the
exposure, users who went on to watch the random video were already seeing 1.27 points
more of its topic than controls, and that gap was growing.

## 2026-11-14: wording of the decision (clarified)

The pre-registered rule returns one of three answers, and it also says an interval
entirely below zero is reported as its own finding. Both apply at once: all three
intervals are inside the half-point margin (the rule says "noise"), and the interval
for the next 10 impressions is entirely below zero. I report both. I did not change
the rule.
