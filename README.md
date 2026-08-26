# Employee Research & Multi-Site Analytics

A real, tested Python project built specifically to close a gap for Siemens
AG's "Working Student Data Analyst (People & Organization)" posting. Its
first listed task is "support the development and adaptation of research
designs for specific user settings" across multiple factory locations,
followed by coordinating multi-site data collection, cleaning/aggregating
the resulting data for statistical analysis, and producing documentation
and dashboards from it. Nothing in my prior project portfolio touched
research-instrument design or organizational/HR-style survey data — this
project builds that skill concretely, end to end, rather than claiming it
without evidence.

The project does four things, chained together: (1) designs a real
multi-construct survey instrument, (2) simulates realistic multi-site data
collection with the data-quality problems real survey data actually has,
(3) cleans the data and scores it with a checked internal-consistency
reliability measure (Cronbach's alpha), and (4) tests whether the sites are
actually statistically and practically comparable before reporting any
pooled or cross-site number.

## What this is (read before citing anywhere)

**There is no real Siemens, employee, or HR data here.** `src/collect_data.py`
generates synthetic Likert-scale survey responses across four fictional
factory sites (Munich, Regensburg, Erlangen, Amberg), not real employee
records, which I have no access to and no authority to collect. I have no
prior professional experience running organizational research or employee
surveys — this project is evidence of research-design and statistical
capability, not of on-the-job HR analytics experience.

## The research instrument (`src/instrument.py`)

Three named constructs, each measured with 4 Likert items (1–5 scale), not
one ad-hoc question per topic — a single item is noisy and can't be
checked for internal consistency, while a multi-item construct can be, via
Cronbach's alpha:

- **Psychological safety** — adapted from the well-established Edmondson
  (1999) psychological-safety construct (publicly documented survey
  research, not invented from scratch).
- **Workload sustainability** — whether current workload is manageable
  without burnout risk.
- **Manager support** — whether employees feel supported and developed by
  their direct manager.

Each construct includes exactly one **reverse-coded item** (e.g. "It is
difficult to ask others on my team for help.") — a standard survey-design
technique: a respondent who straight-lines (picks the same answer for
every item without reading them) will contradict themselves on a
reverse-coded item, which is what makes straightlining detectable at all
(see `src/quality_checks.py`).

## Honest finding #1: my first data simulator produced statistically useless survey data — items didn't correlate with each other at all

My first version of `collect_data.py` sampled every item's response
independently around its site's mean. Running Cronbach's alpha on the
result gave numbers like **-0.06** for psychological safety — worse than
zero, meaning the four items measuring "the same construct" had no
coherent relationship to each other at all. If I had shipped that, the
entire premise of the project (that a multi-item score is a meaningful,
checkable measurement) would have been false.

I didn't average this away or drop the reliability check — I diagnosed it:

- Inspected the item correlation matrix directly: every off-diagonal
  correlation was near zero.
- Ran a standalone sanity-check script confirming the actual cause:
  independent per-item noise around a group mean has no shared signal for
  Cronbach's alpha to detect, by construction.

**The fix:** added `_sample_respondent_true_score()` — each respondent now
gets one latent "true" construct score per construct, drawn around their
site's mean, and every item for that respondent on that construct is a
noisy measurement of that *same* shared value. This is the psychometrically
standard way to simulate a real underlying construct, and it's the reason
multiple items measuring one thing correlate with each other in real
survey data.

## Honest finding #2: fixing finding #1 wasn't enough — reverse-coded items were anti-correlating with the rest of their construct

After the fix above, alpha was *still* near zero. Inspecting the item
correlation matrix again showed the reverse-coded item in each construct
(e.g. `PS3`) correlating **negatively** (-0.27 to -0.35) with its
construct's other three items, instead of positively.

Root cause: the code was sampling a raw response around the *un-inverted*
true score for every item, reverse-coded or not, and only reverse-coding
the *sampled value* afterward. That double-inverts the signal for reverse
items relative to normal items — a respondent with genuinely high
psychological safety would get a high raw sample on the reverse item
"It's difficult to ask for help", which then gets reverse-coded *down*,
backwards from what a real respondent would actually do (someone who
feels safe should *disagree* with that statement, giving a *low* raw
rating, which reverse-codes back *up* to a high construct-aligned score).

**The fix, and the more important lesson:** two changes, together.

1. In `collect_data.py`, reverse-coded items now sample their raw response
   around the *inverted* true score, so the stored raw value is what a
   real respondent would actually circle on the reverse-worded item.
2. Reverse-coding itself moved out of data collection entirely and into
   `scoring.py`'s `apply_reverse_coding()`, applied at analysis time. This
   is also the *psychometrically correct* practice independent of this
   bug: raw survey data should never be destructively transformed at
   collection time, because it makes a coding mistake require
   re-collecting data instead of just re-running analysis.

After both fixes: Cronbach's alpha reached **0.82 / 0.85 / 0.82** across
the three constructs (all in the "good" reliability band), and
`tests/test_scoring.py` now includes a direct regression test
(`TestReverseItemsCorrelatePositively`) asserting every reverse-coded item
correlates *positively*, not negatively, with its construct's other items
— specifically so this bug can't silently reappear.

## Data-quality handling (`src/quality_checks.py`)

- **Straightlining detection**: flags a respondent only if *all* their
  non-missing responses are identical *and* span both a reverse-coded and
  a non-reverse-coded item — a genuine respondent with one true neutral
  opinion would still differ on a reverse item if they're actually reading
  it, so this is a specific signal, not just "gave a lot of 3s."
- **Item-level missingness** (~3% injected) is surfaced per item, not
  silently imputed.
- **Respondent completion rate** and a configurable minimum-completion
  cutoff (default 75%) are applied *before* scoring, so excluded
  respondents never quietly influence a construct score.

## Cross-site comparability (`src/comparability.py`)

Pooling multi-site survey data into one overall number is only valid if
scores are actually comparable across sites. This project does two
things, at two different levels of rigor, and is explicit about which is
which:

1. A real one-way ANOVA (`scipy.stats.f_oneway`) per construct, testing
   whether site means genuinely differ — the right question to ask
   *before* pooling, not skipped in favor of jumping straight to an
   overall mean.
2. A separate, disclosed practical-significance check
   (`pairwise_site_gaps`) flagging site pairs whose mean differs by at
   least 0.4 points on the 1–5 scale, independent of statistical
   significance — because a tiny but "significant" gap in a large sample
   isn't necessarily operationally important, and a large gap in a
   smaller/noisier sample might not reach significance but is still worth
   a human looking at.

**What this project explicitly does NOT implement**, stated directly
rather than silently substituted with the simpler ANOVA above: full
*measurement invariance* testing (configural/metric/scalar invariance via
multi-group confirmatory factor analysis) — the gold-standard check for
whether a survey scale means the same thing to respondents across
different groups *before* comparing their means at all. That technique
needs a larger per-site sample and CFA tooling beyond this project's
scope. The full text of this disclosure lives in code as
`comparability.MEASUREMENT_INVARIANCE_NOTE` and is printed by
`run_pipeline.py`, so it's never separated from the numbers it qualifies.

## Does the pipeline actually recover the injected ground truth?

The simulator deliberately configures Amberg as an outlier site with
genuinely lower manager-support levels (true mean 2.7, vs. 3.5–4.0
elsewhere). `tests/test_collect_data.py::TestSiteMeanRecovery` and
`tests/test_comparability.py::TestOnRealGeneratedData` both run the full
collect → clean → score → compare pipeline end to end and confirm: Amberg
is correctly recovered as the lowest-scoring site, the gap is
statistically significant (ANOVA), and it clears the practical-significance
threshold (actual recovered gap in a live `run_pipeline.py` run: **1.1
points** against the next-closest site). This is the test that actually
matters — individual function unit tests alone wouldn't confirm the
project's core claim, that real between-site differences survive the full
pipeline instead of being averaged away.

## What this doesn't demonstrate

- No real HR/People & Organization team experience, no live employee
  survey ever fielded, no IRB/ethics review process (not applicable to
  synthetic data, but a real consideration for genuine employee research).
- No multi-group CFA / measurement invariance testing (see above,
  disclosed explicitly rather than silently skipped).
- Response and employee-count figures per site are illustrative synthetic
  parameters, not derived from any real organization's headcount.

## Running it

```bash
pip install pandas numpy scipy pytest
pytest tests/ -v          # 54 tests
python run_pipeline.py    # end-to-end demo: collect -> clean -> score -> compare
```
