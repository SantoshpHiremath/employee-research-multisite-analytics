"""
Turns cleaned, RAW item-level responses into construct scores, and checks
internal consistency (Cronbach's alpha) per construct -- the standard
psychometric check that a construct's items are actually measuring one
coherent underlying thing, not an arbitrary grouping. This is the
methodological justification for treating "psychological_safety" as a
single number at all.

Raw survey responses are stored exactly as respondents gave them
(including reverse-worded items, un-adjusted) -- never destructively
transformed at collection time. Reverse-coding is applied here, at
analysis time, via `apply_reverse_coding`, which is the standard,
correct practice: it keeps the raw data auditable and means a coding
mistake can be fixed by re-running analysis, not by re-collecting data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.instrument import CONSTRUCTS, SURVEY_ITEMS, get_construct, reverse_code


def apply_reverse_coding(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of the response dataframe with every reverse-coded
    item's raw values converted to their construct-aligned direction
    (i.e. after this, a HIGH value always means MORE of the construct,
    for every item). Missing values pass through unchanged.
    """
    result = df.copy()
    for item in SURVEY_ITEMS:
        if item.reverse_coded and item.item_id in result.columns:
            result[item.item_id] = result[item.item_id].apply(
                lambda v: reverse_code(int(v)) if pd.notna(v) else v
            )
    return result


def cronbachs_alpha(item_matrix: pd.DataFrame) -> float:
    """Standard Cronbach's alpha formula:

        alpha = (k / (k-1)) * (1 - sum(item_variances) / total_variance)

    where k is the number of items. This is the textbook formula (not
    invented) for internal-consistency reliability. IMPORTANT: the
    item_matrix passed in must already be reverse-coding-adjusted (see
    apply_reverse_coding) -- computing alpha on raw, un-adjusted items
    was the exact mistake that originally produced a near-zero/negative
    alpha during this project's development (see README's "Results"
    section) before being traced to a reverse-coding bug, not a
    genuinely unreliable construct. Rows with any missing value are
    dropped for this calculation (listwise deletion) -- a standard,
    simple choice; more sophisticated missing-data handling
    exists but isn't warranted given this dataset's low (~3%) item-level
    missingness.
    """
    complete = item_matrix.dropna()
    k = item_matrix.shape[1]
    if k < 2 or len(complete) < 2:
        raise ValueError("Need at least 2 items and 2 complete respondents for alpha")

    item_variances = complete.var(axis=0, ddof=1)
    total_scores = complete.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    if total_variance == 0:
        return 0.0

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def score_construct(df_reverse_coded: pd.DataFrame, construct_name: str) -> pd.Series:
    """Per-respondent mean score for a construct, from an ALREADY
    reverse-coding-adjusted dataframe (see apply_reverse_coding) -- skips
    missing items rather than requiring full completion on every item.
    """
    construct = get_construct(construct_name)
    return df_reverse_coded[construct.item_ids].mean(axis=1, skipna=True)


def score_all_constructs(df: pd.DataFrame) -> pd.DataFrame:
    """Applies reverse-coding, then adds one score column per construct."""
    adjusted = apply_reverse_coding(df)
    result = adjusted.copy()
    for construct in CONSTRUCTS:
        result[f"{construct.name}_score"] = score_construct(adjusted, construct.name)
    return result


def construct_reliability_report(df: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha for every construct (computed on reverse-coding-
    adjusted items -- see cronbachs_alpha docstring), with a
    plain-language reliability band (below 0.6 = poor, 0.6-0.7 =
    questionable, 0.7-0.8 = acceptable, above 0.8 = good) -- the
    widely-cited, standard interpretation bands (Nunnally, 1978 / George
    & Mallery, 2003), not invented thresholds.
    """
    adjusted = apply_reverse_coding(df)
    rows = []
    for construct in CONSTRUCTS:
        alpha = cronbachs_alpha(adjusted[construct.item_ids])
        if alpha < 0.6:
            band = "poor"
        elif alpha < 0.7:
            band = "questionable"
        elif alpha < 0.8:
            band = "acceptable"
        else:
            band = "good"
        rows.append({"construct": construct.name, "cronbachs_alpha": alpha, "reliability_band": band})
    return pd.DataFrame(rows)
