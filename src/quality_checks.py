"""
Data-quality checks for the raw survey responses: straightlining
detection (via reverse-coded item consistency) and missingness handling
-- the cleaning step of the workflow, done with a defensible method
rather than an arbitrary rule.
"""
from __future__ import annotations

import pandas as pd

from src.instrument import SURVEY_ITEMS, get_construct, CONSTRUCTS


def flag_straightliners(df: pd.DataFrame) -> pd.Series:
    """Flags a respondent as a likely straightliner if ALL of their
    non-missing item responses are identical AND that includes at least
    one reverse-coded and one non-reverse-coded item -- a genuine
    respondent who happens to feel a "3" on everything would still answer
    a reverse-coded item differently if they're reading it, so identical
    responses across both item types is a strong, specific signal, not
    just "gave a lot of neutral answers."
    """
    item_ids = [item.item_id for item in SURVEY_ITEMS]
    reverse_ids = {item.item_id for item in SURVEY_ITEMS if item.reverse_coded}
    normal_ids = {item.item_id for item in SURVEY_ITEMS if not item.reverse_coded}

    flags = []
    for _, row in df.iterrows():
        responses = row[item_ids].dropna()
        answered_reverse = set(responses.index) & reverse_ids
        answered_normal = set(responses.index) & normal_ids
        is_flagged = (
            len(responses) > 1
            and responses.nunique() == 1
            and len(answered_reverse) > 0
            and len(answered_normal) > 0
        )
        flags.append(is_flagged)
    return pd.Series(flags, index=df.index, name="is_likely_straightliner")


def item_missingness_rate(df: pd.DataFrame) -> pd.Series:
    """Per-item fraction of missing responses -- surfaced explicitly so a
    high-missingness item can be flagged for redesign, rather than
    silently imputed away.
    """
    item_ids = [item.item_id for item in SURVEY_ITEMS]
    return df[item_ids].isna().mean()


def respondent_completion_rate(df: pd.DataFrame) -> pd.Series:
    """Per-respondent fraction of items answered."""
    item_ids = [item.item_id for item in SURVEY_ITEMS]
    return df[item_ids].notna().mean(axis=1)


def clean_responses(df: pd.DataFrame, min_completion_rate: float = 0.75) -> pd.DataFrame:
    """Removes straightliners and respondents below a minimum completion
    rate -- both adjustable exclusion criteria, applied
    BEFORE scoring, not after (so excluded respondents never silently
    influence construct scores).
    """
    straightliners = flag_straightliners(df)
    completion = respondent_completion_rate(df)
    keep_mask = (~straightliners) & (completion >= min_completion_rate)
    return df.loc[keep_mask].reset_index(drop=True)
