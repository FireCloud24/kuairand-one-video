# Related work, in plain English

This note says what earlier work did and where my project sits. I summarise the
filter-bubble audits from their abstracts, and I say so where it matters.

## The dataset

**KuaiRand** (Gao, Li, Zhang, Chen, Li, Lei, Jiang and He, CIKM 2022, arXiv 2208.08696).
For about two weeks in April and May 2022, Kuaishou replaced a small share of the
videos in real users' feeds with videos drawn at random from a fixed pool of 7,583.
The released logs mark which impressions were random. They also keep the ordinary
feed for the same users, twelve kinds of feedback (clicks, likes, watch time and so
on), and features for users and videos.

The authors built it for a different job from mine. A normal log only shows what the
recommender chose to show, so any model trained on it inherits the recommender's
taste. Random impressions do not have that problem, so they make a fair test set for
"debiased" recommendation models. Most papers that use KuaiRand use it this way.

I use the random impressions as a natural experiment instead. Chance decided the
topic of each random video. So I can ask what the feed did next, and compare people
who happened to get topic T with people who happened to get something else.

## Filter-bubble audits of short-video apps

**Sock-puppet audits.** Boeker and Urman, "An Empirical Investigation of
Personalization Factors on TikTok" (WWW 2022, arXiv 2201.12271), ran automated
accounts on TikTok and changed one thing at a time: location, language, likes,
follows, and how long a video was watched. They then measured how far each account's
feed moved away from a twin account. Later audits use the same idea on narrower
questions, for example Pecher et al., "Algorithmic Audit of Personalisation Drift in
Polarising Topics on TikTok" (arXiv 2603.20723, 2026). These audits control the
"user" completely, which is their strength. Their weakness is that a bot with no
history is a strange user, and it usually acts on a topic many times in a row.

**Log studies.** Sukiennik, Gao and Li, "Uncovering the Deep Filter Bubble: Narrow
Exposure in Short-Video Recommendation" (WWW 2024, arXiv 2403.04511), followed real
users of a large Chinese short-video platform for a year and tracked how narrow each
user's exposure was within a top-level category. This is real behaviour at scale. But
it is observational: a user who sees more of a topic may simply like it more, so the
log alone cannot say the recommender caused the narrowing.

**Simulated users.** PersonaAct (Zhao, Yang, Yin, Wang, Chen, Su and Cheng, arXiv
2601.22547, 2026) trains multimodal agents on real behaviour traces, gives each one a
persona, and lets the agents browse short-video platforms. The authors measure how
narrow the feed gets (breadth) and how hard it is to get out (depth), and report that
feeds narrow as the agents interact. This scales well and avoids privacy problems.
The cost is that the conclusions depend on how closely an agent behaves like a person.

## Where this project sits

Each approach above gives up something. Sock puppets and agents are controlled but
are not real users. Log studies have real users but no control group.

KuaiRand offers a small piece of both: real users, a real production recommender, and
an exposure whose topic was set by chance. The price is that the treatment is tiny.
It is one video, in one slot, that the user may not even have watched. The audits
above study repeated, deliberate engagement. I study the smallest nudge there is.

That is why both of my explanations are believable in advance. A recommender that
reacts quickly to implicit feedback could move after a single view. Or one video
could be lost among the hundreds a user sees in a day. I think knowing which one
holds is useful for the audit literature, because it says how much of a "bubble" can
start from an accident.
