import numpy as np
import pandas as pd
import pytest

from src.collect_data import generate_responses
from src.instrument import SURVEY_ITEMS, CONSTRUCTS, get_construct
from src.quality_checks import clean_responses
from src.scoring import (
    apply_reverse_coding, cronbachs_alpha, score_construct,
    score_all_constructs, construct_reliability_report,
)


@pytest.fixture(scope="module")
def cleaned_df():
    raw = generate_responses(seed=42)
    return clean_responses(raw)


class TestApplyReverseCoding:
    def test_reverse_item_values_are_flipped(self):
        construct = CONSTRUCTS[0]
        reverse_item_ids = [
            iid for iid in construct.item_ids
            if any(i.item_id == iid and i.reverse_coded for i in SURVEY_ITEMS)
        ]
        assert reverse_item_ids, "fixture assumption: construct has a reverse item"
        reverse_id = reverse_item_ids[0]

        df = pd.DataFrame({"respondent_id": [1, 2], "site": ["Munich", "Munich"], reverse_id: [1, 5]})
        adjusted = apply_reverse_coding(df)
        # 1 -> 5, 5 -> 1 under the 1-5 Likert reverse mapping
        assert adjusted[reverse_id].tolist() == [5, 1]

    def test_non_reverse_item_values_are_untouched(self):
        normal_item = next(i for i in SURVEY_ITEMS if not i.reverse_coded)
        df = pd.DataFrame({"respondent_id": [1, 2], "site": ["Munich", "Munich"], normal_item.item_id: [1, 5]})
        adjusted = apply_reverse_coding(df)
        assert adjusted[normal_item.item_id].tolist() == [1, 5]

    def test_missing_values_pass_through_unchanged(self):
        reverse_item = next(i for i in SURVEY_ITEMS if i.reverse_coded)
        df = pd.DataFrame({"respondent_id": [1], "site": ["Munich"], reverse_item.item_id: [np.nan]})
        adjusted = apply_reverse_coding(df)
        assert pd.isna(adjusted[reverse_item.item_id].iloc[0])

    def test_does_not_mutate_input_dataframe(self):
        reverse_item = next(i for i in SURVEY_ITEMS if i.reverse_coded)
        df = pd.DataFrame({"respondent_id": [1], "site": ["Munich"], reverse_item.item_id: [1]})
        original = df.copy()
        apply_reverse_coding(df)
        pd.testing.assert_frame_equal(df, original)


class TestCronbachsAlpha:
    def test_matches_hand_computed_formula(self):
        # Hand-built 4-item matrix, 6 respondents, chosen so items co-vary
        # (simulating a real coherent construct) -- alpha computed here via
        # the textbook formula independently of the function under test.
        item_matrix = pd.DataFrame({
            "i1": [5, 4, 3, 2, 1, 4],
            "i2": [5, 4, 4, 2, 1, 3],
            "i3": [4, 4, 3, 1, 2, 4],
            "i4": [5, 3, 3, 2, 1, 4],
        })
        k = item_matrix.shape[1]
        item_variances = item_matrix.var(axis=0, ddof=1)
        total_variance = item_matrix.sum(axis=1).var(ddof=1)
        expected = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

        result = cronbachs_alpha(item_matrix)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_perfectly_correlated_items_give_alpha_near_one(self):
        base = pd.Series([1, 2, 3, 4, 5, 3, 2, 4])
        item_matrix = pd.DataFrame({"i1": base, "i2": base, "i3": base})
        assert cronbachs_alpha(item_matrix) == pytest.approx(1.0, abs=1e-9)

    def test_raises_with_fewer_than_two_items(self):
        item_matrix = pd.DataFrame({"i1": [1, 2, 3, 4]})
        with pytest.raises(ValueError):
            cronbachs_alpha(item_matrix)

    def test_raises_with_fewer_than_two_complete_respondents(self):
        item_matrix = pd.DataFrame({"i1": [1, np.nan, np.nan], "i2": [2, np.nan, np.nan]})
        with pytest.raises(ValueError):
            cronbachs_alpha(item_matrix)

    def test_real_generated_data_reliability_is_at_least_acceptable(self, cleaned_df):
        """Regression test for both fixed bugs: on real generated (cleaned)
        data, every construct's reverse-coding-adjusted alpha should land
        in the 'acceptable' or 'good' band (>=0.7). Before the two bugs
        were fixed, these came out near zero or negative.
        """
        adjusted = apply_reverse_coding(cleaned_df)
        for construct in CONSTRUCTS:
            alpha = cronbachs_alpha(adjusted[construct.item_ids].dropna())
            assert alpha >= 0.7, f"{construct.name} alpha={alpha} below acceptable band"


class TestReverseItemsCorrelatePositively:
    """Direct regression test for Bug 2 (the double-inversion bug): after
    apply_reverse_coding, a construct's reverse-coded item(s) must
    correlate POSITIVELY with its normal items, not negatively. This is
    exactly the symptom that exposed the bug during development (PS3 was
    correlating at -0.27 to -0.35 with the other psychological_safety
    items before the fix).
    """

    def test_reverse_item_correlates_positively_with_normal_items(self, cleaned_df):
        adjusted = apply_reverse_coding(cleaned_df)
        for construct in CONSTRUCTS:
            items = [i for i in SURVEY_ITEMS if i.item_id in construct.item_ids]
            reverse_ids = [i.item_id for i in items if i.reverse_coded]
            normal_ids = [i.item_id for i in items if not i.reverse_coded]
            assert reverse_ids and normal_ids, "fixture assumption"

            complete = adjusted[construct.item_ids].dropna()
            for rid in reverse_ids:
                for nid in normal_ids:
                    corr = complete[rid].corr(complete[nid])
                    assert corr > 0, (
                        f"{construct.name}: reverse item {rid} correlates "
                        f"{corr:.3f} with normal item {nid} (expected positive)"
                    )


class TestScoreConstruct:
    def test_score_is_within_likert_range(self, cleaned_df):
        adjusted = apply_reverse_coding(cleaned_df)
        for construct in CONSTRUCTS:
            scores = score_construct(adjusted, construct.name)
            valid = scores.dropna()
            assert (valid >= 1).all()
            assert (valid <= 5).all()

    def test_score_is_mean_of_construct_items(self, cleaned_df):
        adjusted = apply_reverse_coding(cleaned_df)
        construct = get_construct("psychological_safety")
        scores = score_construct(adjusted, construct.name)
        manual = adjusted[construct.item_ids].mean(axis=1, skipna=True)
        pd.testing.assert_series_equal(scores, manual, check_names=False)


class TestScoreAllConstructs:
    def test_adds_one_score_column_per_construct(self, cleaned_df):
        result = score_all_constructs(cleaned_df)
        for construct in CONSTRUCTS:
            assert f"{construct.name}_score" in result.columns

    def test_row_count_is_unchanged(self, cleaned_df):
        result = score_all_constructs(cleaned_df)
        assert len(result) == len(cleaned_df)

    def test_applies_reverse_coding_internally(self, cleaned_df):
        """Guards against someone accidentally scoring raw (un-adjusted)
        items directly -- scores from score_all_constructs must match
        scoring the explicitly-adjusted frame.
        """
        result = score_all_constructs(cleaned_df)
        adjusted = apply_reverse_coding(cleaned_df)
        for construct in CONSTRUCTS:
            expected = score_construct(adjusted, construct.name)
            pd.testing.assert_series_equal(
                result[f"{construct.name}_score"], expected,
                check_names=False,
            )


class TestConstructReliabilityReport:
    def test_returns_one_row_per_construct(self, cleaned_df):
        report = construct_reliability_report(cleaned_df)
        assert len(report) == len(CONSTRUCTS)
        assert set(report["construct"]) == {c.name for c in CONSTRUCTS}

    def test_reliability_band_matches_alpha_thresholds(self, cleaned_df):
        report = construct_reliability_report(cleaned_df)
        for _, row in report.iterrows():
            alpha = row["cronbachs_alpha"]
            band = row["reliability_band"]
            if alpha < 0.6:
                assert band == "poor"
            elif alpha < 0.7:
                assert band == "questionable"
            elif alpha < 0.8:
                assert band == "acceptable"
            else:
                assert band == "good"

    def test_real_data_bands_are_acceptable_or_good(self, cleaned_df):
        report = construct_reliability_report(cleaned_df)
        assert report["reliability_band"].isin(["acceptable", "good"]).all()
