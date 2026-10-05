"""
Simulates multi-site survey data collection: generates realistic (but
synthetic) Likert
responses across several factory sites, with deliberately injected
real-world data-collection problems:

- Different sites have genuinely different underlying construct levels
  (a real, expected finding in multi-site organizational research, not
  noise to be explained away).
- A minority of respondents straight-line (pick the same response for
  every item without reading them) -- a well-documented survey data-
  quality problem.
- Sites have different response rates and different total employee
  counts, so naive unweighted pooling would over-represent large sites.
- A small fraction of responses have item-level missingness (a
  respondent skips one or more items), which is realistic and must be
  handled explicitly rather than silently dropped or zero-filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.instrument import SURVEY_ITEMS, SITES, LIKERT_MIN, LIKERT_MAX

# Each site's true underlying mean construct levels (1-5 scale) -- these
# are the "ground truth" the simulator is built around, so tests can
# check the analysis pipeline actually recovers them.
SITE_TRUE_MEANS = {
    "Munich": {"psychological_safety": 3.8, "workload_sustainability": 3.2, "manager_support": 3.9},
    "Regensburg": {"psychological_safety": 3.6, "workload_sustainability": 2.6, "manager_support": 3.5},
    "Erlangen": {"psychological_safety": 4.0, "workload_sustainability": 3.5, "manager_support": 4.0},
    # Amberg is deliberately the outlier site -- genuinely different
    # (lower) manager-support levels, which the analysis should surface,
    # not smooth over as if all sites were the same.
    "Amberg": {"psychological_safety": 3.4, "workload_sustainability": 3.0, "manager_support": 2.7},
}

SITE_EMPLOYEE_COUNTS = {"Munich": 450, "Regensburg": 210, "Erlangen": 180, "Amberg": 95}
SITE_RESPONSE_RATES = {"Munich": 0.55, "Regensburg": 0.62, "Erlangen": 0.70, "Amberg": 0.48}

STRAIGHTLINE_RATE = 0.06  # ~6% of respondents straight-line, a realistic real-world figure
ITEM_MISSINGNESS_RATE = 0.03  # ~3% chance any individual item is skipped


def _sample_response(true_score: float, rng: np.random.Generator, item_noise_std: float = 0.55) -> int:
    """Samples one Likert response around a RESPONDENT-LEVEL true
    construct score (not just the site mean -- see
    _sample_respondent_true_score), rounded and clipped to the valid 1-5
    range. item_noise_std=0.55 is deliberately smaller than the earlier
    (buggy) version's 0.9: see the "Results" section in README --
    the first version sampled every item independently around the SITE
    mean with no shared respondent-level signal, which made items
    essentially uncorrelated (Cronbach's alpha near zero). Modeling a
    shared per-respondent latent score, with each item as a noisy
    measurement of it, is the psychometrically correct way to simulate a
    real underlying construct, and produces realistic item
    intercorrelations (confirmed in tests/test_scoring.py).
    """
    raw = rng.normal(true_score, item_noise_std)
    return int(np.clip(round(raw), LIKERT_MIN, LIKERT_MAX))


def _sample_respondent_true_score(site_mean: float, rng: np.random.Generator, between_person_std: float = 0.6) -> float:
    """Each respondent has their own true construct level, drawn around
    their site's mean -- this is what every item for that respondent (on
    that construct) is a noisy measurement OF, which is what makes items
    correlate with each other for the same person (and is the entire
    point of a multi-item construct).
    """
    return rng.normal(site_mean, between_person_std)


def generate_responses(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    respondent_id = 0

    for site in SITES:
        n_respondents = round(SITE_EMPLOYEE_COUNTS[site] * SITE_RESPONSE_RATES[site])
        for _ in range(n_respondents):
            respondent_id += 1
            is_straightliner = rng.random() < STRAIGHTLINE_RATE
            straightline_value = int(rng.integers(LIKERT_MIN, LIKERT_MAX + 1)) if is_straightliner else None

            # One latent true score PER CONSTRUCT PER RESPONDENT, drawn
            # once per respondent (not once per item) -- this is what
            # makes that respondent's items on the same construct
            # correlate with each other, which is the entire reason a
            # multi-item construct is checkable via Cronbach's alpha.
            respondent_true_scores = {
                construct_name: _sample_respondent_true_score(site_mean, rng)
                for construct_name, site_mean in SITE_TRUE_MEANS[site].items()
            }

            row = {"respondent_id": respondent_id, "site": site}
            for item in SURVEY_ITEMS:
                if rng.random() < ITEM_MISSINGNESS_RATE:
                    row[item.item_id] = np.nan
                    continue

                if is_straightliner:
                    # A straightliner picks ONE value and gives it for
                    # every item, ignoring reverse-coding -- exactly the
                    # behavior that makes reverse-coded items useful for
                    # detection (see quality_checks.py).
                    row[item.item_id] = straightline_value
                else:
                    true_score = respondent_true_scores[item.construct]
                    # Reverse-coded items: a respondent with a HIGH true
                    # construct score should give a LOW raw rating on a
                    # reverse-worded item (e.g. high psychological safety
                    # -> disagrees that "it's difficult to ask for help").
                    # So for reverse items we sample around the INVERTED
                    # true score, producing a raw response that -- once
                    # reverse-coded during scoring -- correctly recovers
                    # the shared latent construct signal. Sampling around
                    # the un-inverted true_score and reverse-coding
                    # AFTERWARD (the original, buggy version) flips the
                    # sign twice for reverse items relative to normal
                    # items, making them anti-correlate instead of
                    # correlate -- this was a real bug, caught via
                    # Cronbach's alpha coming out near zero/negative (see
                    # README's "Results" section).
                    if item.reverse_coded:
                        inverted_true_score = LIKERT_MAX + LIKERT_MIN - true_score
                        raw_response = _sample_response(inverted_true_score, rng)
                    else:
                        raw_response = _sample_response(true_score, rng)
                    row[item.item_id] = raw_response
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = generate_responses()
    df.to_csv("data/survey_responses.csv", index=False)
    print(f"Generated {len(df)} responses across {df['site'].nunique()} sites.")
    print(df["site"].value_counts())


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()
