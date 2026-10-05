# Employee Research & Multi-Site Analytics

A tested Python project for employee-survey research across multiple
factory locations: research-instrument design, multi-site data collection,
cleaning and aggregation for statistical analysis, and cross-site
comparison.

It does four things, chained together: (1) designs a multi-construct survey
instrument, (2) simulates multi-site data collection with the data-quality
problems real survey data has, (3) cleans the data and scores it with a
checked internal-consistency reliability measure (Cronbach's alpha), and
(4) tests whether the sites are statistically and practically comparable
before reporting any pooled or cross-site number.

## Data

The data is synthetic. `src/collect_data.py` generates Likert-scale survey
responses across four fictional factory sites (Munich, Regensburg,
Erlangen, Amberg). Response and employee-count figures per site are
illustrative parameters. The pipeline is built so real survey responses can
replace the simulator.

## What it does

### The research instrument (`src/instrument.py`)

Three named constructs, each measured with 4 Likert items (1–5 scale)
rather than one ad-hoc question per topic. A single item is noisy and can't
be checked for internal consistency, while a multi-item construct can be,
via Cronbach's alpha:

- **Psychological safety** — adapted from the well-established Edmondson
  (1999) psychological-safety construct (publicly documented survey
  research).
- **Workload sustainability** — whether current workload is manageable
  without burnout risk.
- **Manager support** — whether employees feel supported and developed by
  their direct manager.

Each construct includes exactly one **reverse-coded item** (e.g. "It is
difficult to ask others on my team for help."). This is a standard
survey-design technique: a respondent who straight-lines (picks the same
answer for every item without reading them) will contradict themselves on
a reverse-coded item, which is what makes straightlining detectable (see
`src/quality_checks.py`).

### Data-quality handling (`src/quality_checks.py`)

- **Straightlining detection**: flags a respondent only if *all* their
  non-missing responses are identical *and* span both a reverse-coded and a
  non-reverse-coded item. A genuine respondent with one true neutral
  opinion would still differ on a reverse item if they're actually reading
  it, so this is a specific signal, not just "gave a lot of 3s."
- **Item-level missingness** (~3% injected) is surfaced per item, not
  silently imputed.
- **Respondent completion rate** and a configurable minimum-completion
  cutoff (default 75%) are applied *before* scoring, so excluded
  respondents never influence a construct score.

### Cross-site comparability (`src/comparability.py`)

Pooling multi-site survey data into one overall number is only valid if
scores are comparable across sites. The project checks this at two levels:

1. A one-way ANOVA (`scipy.stats.f_oneway`) per construct, testing whether
   site means differ. This is the right question to ask *before* pooling.
2. A practical-significance check (`pairwise_site_gaps`) flagging site
   pairs whose mean differs by at least 0.4 points on the 1–5 scale,
   independent of statistical significance. A tiny but "significant" gap in
   a large sample isn't necessarily operationally important, and a large
   gap in a smaller or noisier sample might not reach significance but is
   still worth a human looking at.

Full *measurement invariance* testing (configural/metric/scalar invariance
via multi-group confirmatory factor analysis) is the natural next step for
checking whether a survey scale means the same thing across groups; it
needs a larger per-site sample and CFA tooling. The scope note lives in
code as `comparability.MEASUREMENT_INVARIANCE_NOTE` and is printed by
`run_pipeline.py` alongside the numbers it qualifies.

## Results

### Recovering the injected ground truth

The simulator configures Amberg as an outlier site with lower
manager-support levels (true mean 2.7, vs. 3.5–4.0 elsewhere).
`tests/test_collect_data.py::TestSiteMeanRecovery` and
`tests/test_comparability.py::TestOnRealGeneratedData` both run the full
collect → clean → score → compare pipeline end to end and confirm that
Amberg is recovered as the lowest-scoring site, the gap is statistically
significant (ANOVA), and it clears the practical-significance threshold
(recovered gap in a live `run_pipeline.py` run: **1.1 points** against the
next-closest site). Real between-site differences survive the full
pipeline instead of being averaged away.

### Making the simulated items behave like a real construct

**Shared respondent-level signal.** My first version of `collect_data.py`
sampled every item's response independently around its site's mean, which
gave Cronbach's alpha values like **-0.06** for psychological safety:
independent per-item noise has no shared signal for alpha to detect. I
inspected the item correlation matrix (every off-diagonal correlation was
near zero) and added `_sample_respondent_true_score()`. Each respondent now
gets one latent "true" construct score per construct, drawn around their
site's mean, and every item for that respondent on that construct is a
noisy measurement of that same shared value. This is the standard
psychometric way to simulate an underlying construct.

**Reverse-coded items.** After that change, alpha was still near zero. The
reverse-coded item in each construct (e.g. `PS3`) correlated **negatively**
(-0.27 to -0.35) with its construct's other three items. The code was
sampling a raw response around the *un-inverted* true score for every item
and only reverse-coding the sampled value afterward, which double-inverts
the signal for reverse items. Two changes fixed it:

1. In `collect_data.py`, reverse-coded items now sample their raw response
   around the *inverted* true score, so the stored raw value is what a real
   respondent would circle on the reverse-worded item.
2. Reverse-coding moved out of data collection and into `scoring.py`'s
   `apply_reverse_coding()`, applied at analysis time. Raw survey data
   should not be destructively transformed at collection time, because a
   coding mistake would then require re-collecting data instead of
   re-running analysis.

After both changes, Cronbach's alpha reached **0.82 / 0.85 / 0.82** across
the three constructs (all in the "good" reliability band), and
`tests/test_scoring.py` includes a regression test
(`TestReverseItemsCorrelatePositively`) asserting every reverse-coded item
correlates *positively* with its construct's other items.

## Tests

54 tests (`pytest tests/ -v`) covering the instrument, data collection,
quality checks, scoring, and comparability modules, plus the end-to-end
ground-truth recovery described above.

## Running it

```bash
pip install pandas numpy scipy pytest
pytest tests/ -v          # 54 tests
python run_pipeline.py    # end-to-end demo: collect -> clean -> score -> compare
```

## Possible extensions

- Multi-group CFA / measurement invariance testing across sites.
- Field the instrument with real employee respondents, including the
  ethics-review process that genuine employee research involves.
- Dashboards and documentation built on the scored and compared output.
