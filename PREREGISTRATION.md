# Pre-registration: can one random video bend your feed?

Written and committed on 14 November 2026, before any outcome was computed. The git
history shows this file arriving before the code that builds outcomes. Any later
change of plan gets a dated entry in `DEVIATIONS.md`.

## 1. Question and the two explanations

After one chance encounter with a topic, does the feed start showing the user more of
that topic, and for how long?

- **The feed bends.** One exposure shifts what the recommender serves next. Users who
  got a random video in topic T see a higher share of T afterwards than users who got
  a random video in another topic.
- **Noise.** One exposure is too small a signal. The difference is close to zero.

## 2. What I had already seen when I wrote this

I want to be exact about this, because a pre-registration is only worth something if
the outcome was unseen.

Seen (all in `results/sanity.md` and `results/design_facts.md`):

- Row, user and video counts for each log, and the dates they cover.
- Which random videos were shown, their topics, and counts per topic.
- The ordinary feed from 8 to 21 April 2022 only, which is before the experiment:
  topic shares, and how noisy a window of 10, 30 or 100 impressions is.
- Structure of the logs: the tab column, the share of rows that share a timestamp, the
  time gaps between rows, and whether rows are in time order.
- User features: registration days and activity level.
- One number that touches behaviour during the experiment: 17.4% of random videos were
  "clicked" (a valid play) and 8.4% got a long view, over all topics together.

Not seen: the topic of any ordinary impression on or after 22 April 2022, and so
nothing about what the feed showed after a random exposure.

## 3. Data

KuaiRand-1K (Gao et al., CIKM 2022): 1,000 users, 43,028 random exposures between 22
April and 8 May 2022, and 11,713,045 ordinary impressions between 8 April and 8 May.
Every user in the dataset received at least 10 random videos; that is the dataset's
own inclusion rule.

Each random video replaced one video in a feed request and was drawn uniformly from a
fixed pool, whatever the user's taste. That is what makes the comparison below fair.

## 4. Definitions

**Topic.** A video's topic is its first-level category in the authors' supplementary
file `kuairand_video_categories.csv`. Each video has one. Videos with no category count
as "no topic".

**Target topics.** The first-level categories with at least 1,000 random exposures.
There are 13: Film and TV (39), Games (3), Anime and comics (20), Food (12), Comedy
(9), Sports (7), Short drama (32), Music (2), Dance (1), Celebrities (6), Art (15),
Pets (17) and Beauty (4). They cover 31,926 of the 43,028 random exposures.

**Unit.** One random exposure: a user, a time, and the random video shown.

**Treated and control.** For target topic T, an exposure is treated if the random video
is in T. It is a control if the random video is in any other topic or has no topic. So
every random exposure is in the analysis for every topic, on one side or the other.
The plan asks for controls "at a comparable time". Because the draw was uniform at
every request, all other exposures are comparable by design. I check this in the
balance table instead of assuming it (section 7).

**Ordinary impressions.** Rows of the two standard logs, all tabs. Random videos are
never counted in an outcome.

**Order.** Impressions arrive in feed requests of about six that share one timestamp.
Within a user I order ordinary impressions by timestamp, then by their order in the
file. For an exposure at time t:

- "after" means ordinary impressions with a timestamp later than t;
- "before" means ordinary impressions with a timestamp earlier than t;
- impressions with the same timestamp are the same feed request. The recommender chose
  them before the user saw the random video, so they are in neither window. I report
  them on their own as a check that should show no effect.

**Windows.** The next 10, the next 30 and the next 100 ordinary impressions after the
exposure. They are cumulative: the 30 window contains the 10 window. They may run into
later days. If a window is cut by the end of the log, I use what is there.

**Outcome.** For topic T and window k: the share of the user's next k ordinary
impressions that are in T.

**Baseline.** The share of T in the user's last 100 ordinary impressions before the
exposure (fewer if fewer exist).

## 5. Estimator

For each topic T and window k:

> effect(T, k) = mean of (outcome − baseline) over treated exposures
> minus mean of (outcome − baseline) over control exposures.

Subtracting the baseline does not change what is estimated, because the draw was
random. It removes the differences in taste between users, which makes the estimate
less noisy. I also report the plain difference in outcomes without the baseline.

An exposure enters this estimator if it has at least one ordinary impression before
and at least one after. I report how many do not.

**Pooled effect.** The average of effect(T, k) over the 13 topics, weighted by each
topic's number of treated exposures. This is the headline number: the change in the
share of a topic after one random video in that topic, in percentage points.

Users get many random videos (the median is 22), so windows overlap and a control
exposure may follow an earlier treated one. Randomisation spreads this evenly over
both groups. What I estimate is therefore the effect of one more exposure to T,
on top of whatever else the user was shown.

## 6. Intervals and the decision rule

**Intervals.** Percentile bootstrap that resamples users, with all of a user's
exposures kept together. 2,000 resamples, seed 20261114. Reported intervals are 95%.

**Primary test.** The three pooled effects (windows 10, 30, 100). For the decision I
use 98.33% intervals, so that the three together keep a 5% error rate.

- **The feed bends** if at least one of the three intervals lies entirely above zero.
- **Noise** if all three intervals lie entirely inside ±0.5 percentage points. Half a
  point is one extra video in every 200. The average target topic has about 4% of the
  feed, so half a point is about an eighth of its usual share.
- **Undecided** otherwise. I will say so plainly.

An interval entirely below zero would mean the feed moves away from the topic. I would
report that as its own finding.

Per-topic effects are reported with 95% intervals and no correction. They are there to
show the spread, and I will not single out one topic as a finding.

## 7. Balance check (before any outcome)

For each topic, and pooled the same way as the effect, I compare treated and control
exposures on things fixed before the exposure: the baseline share of T, the day of the
experiment, the hour of the day, the number of ordinary impressions the user had before
the exposure, and the user's days since registration. I report standardised mean
differences with bootstrap intervals.

If the pooled difference in baseline share is larger than 0.05 standard deviations, I
stop and work out why before computing outcomes.

## 8. Event-study figure

Pooled treated-minus-control difference in topic share, with 95% intervals, in bins of
10 ordinary impressions: ten bins before the exposure (−100 to −1), the same-request
bin, and ten bins after (+1 to +100). Bins use the plain share with no baseline
subtracted, so the bins before the exposure are a visible placebo. They should sit at
zero.

## 9. Dropout

Users who leave after the random video stay in the analysis.

- Windows are counted in impressions, with no time limit. A user who leaves and comes
  back the next day still has an outcome.
- I report, treated against control, the share of exposures where the user "left": no
  ordinary impression in a later request within 30 minutes. This tests whether a random
  video in T makes people leave more or less often.
- I report the share of exposures with no later ordinary impression at all before the
  log ends. These cannot have an outcome. If that share differs between treated and
  control by more than one percentage point, I will add bounds to the main result.
- As a check, I repeat the main estimate counting only impressions within 60 minutes of
  the exposure.

## 10. Heterogeneity (exploratory)

Same estimator within subgroups, all three windows, 95% intervals, no correction.

- **New against heavy users.** New: registered for 365 days or fewer (144 users).
  Heavy: the top third of users by ordinary impressions from 8 to 21 April (more than
  5,175). Light: the bottom third (fewer than 2,130). The dataset has no truly new
  users, because of its inclusion rule, and I will say so.
- **Popular against niche topics.** Popular: the six target topics with the largest
  share of the feed before the experiment (Film and TV, Comedy, Food, Games, Anime and
  comics, Music). Niche: the other seven.
- **Watched against skipped.** Watched means the random video was a valid play
  (`is_click` = 1). I compare treated and control exposures among watched videos, and
  again among skipped ones. I repeat it with long views. Watching is the user's choice
  and is not randomised: people watch what they already like. So this split is
  description only, and I will not present it as an effect of watching.

## 11. Other checks I commit to reporting

- The outcome restricted to tab 1, where 99% of random videos appeared.
- Topics defined from the `tag` column of the main dataset instead of the
  supplementary file (a video is in T if T is one of its tags).
- The plain difference without the baseline.

## 12. Known limits, stated in advance

- Apps fetch the feed a few videos ahead. Some of the "next 10" may have been chosen
  before the user saw the random video. That biases the 10 window towards zero, which
  is one reason I pre-register 30 and 100 as well.
- A video in a later window can only respond to the exposure if the recommender uses
  it at all. I cannot see the recommender, only what it showed.
- 1,000 users. If the intervals are too wide to decide, the next step is the full
  27,285-user dataset. I will keep the estimator as it is.
