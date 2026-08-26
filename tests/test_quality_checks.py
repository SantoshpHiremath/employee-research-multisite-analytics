import numpy as np
import pandas as pd
import pytest

from src.collect_data import generate_responses
from src.instrument import SURVEY_ITEMS
from src.quality_checks import (
    flag_straightliners, item_missingness_rate, respondent_completion_rate,
    clean_responses,
)


@pytest.fixture(scope="module")
def raw_df():
    return generate_responses(seed=42)


class TestFlagStraightliners:
    def test_a_hand_built_straightliner_is_flagged(self):
        """A respondent who gave '3' to every item, including a
        reverse-coded one, is exactly the pattern this check exists to
        catch.
        """
        item_ids = [item.item_id for item in SURVEY_ITEMS]
        row = {"respondent_id": 1, "site": "Munich"}
        row.update({iid: 3 for iid in item_ids})
        df = pd.DataFrame([row])
        flags = flag_straightliners(df)
        assert flags.iloc[0] == True

    def test_a_genuine_varied_respondent_is_not_flagged(self):
        item_ids = [item.item_id for item in SURVEY_ITEMS]
        row = {"respondent_id": 1, "site": "Munich"}
        # varied, realistic-looking responses
        values = [4, 4, 2, 4, 3, 2, 3, 4, 5, 4, 2, 4]
        row.update(dict(zip(item_ids, values)))
        df = pd.DataFrame([row])
        flags = flag_straightliners(df)
        assert flags.iloc[0] == False

    def test_realistic_generated_data_has_some_flagged(self, raw_df):
        """Not asserting an exact count (that would be seed-fragile) --
        just that the mechanism finds SOME of the straightliners the
        simulator deliberately injects.
        """
        flags = flag_straightliners(raw_df)
        assert flags.sum() > 0
        assert flags.sum() < len(raw_df) * 0.2  # shouldn't over-flag more than ~20%


class TestMissingnessAndCompletion:
    def test_item_missingness_rate_is_between_zero_and_one(self, raw_df):
        rates = item_missingness_rate(raw_df)
        assert (rates >= 0).all()
        assert (rates <= 1).all()

    def test_respondent_completion_rate_matches_manual_calculation(self, raw_df):
        item_ids = [item.item_id for item in SURVEY_ITEMS]
        completion = respondent_completion_rate(raw_df)
        manual = raw_df[item_ids].notna().mean(axis=1)
        pd.testing.assert_series_equal(completion, manual, check_names=False)


class TestCleanResponses:
    def test_removes_flagged_straightliners(self, raw_df):
        cleaned = clean_responses(raw_df)
        remaining_flags = flag_straightliners(cleaned)
        assert remaining_flags.sum() == 0

    def test_removes_respondents_below_completion_threshold(self, raw_df):
        cleaned = clean_responses(raw_df, min_completion_rate=0.75)
        completion = respondent_completion_rate(cleaned)
        assert (completion >= 0.75).all()

    def test_never_returns_more_rows_than_input(self, raw_df):
        cleaned = clean_responses(raw_df)
        assert len(cleaned) <= len(raw_df)
