import pandas as pd
import pytest

from src.comparability import (
    compare_sites_anova, compare_all_constructs, pairwise_site_gaps,
    MEASUREMENT_INVARIANCE_NOTE,
)


def _controlled_df():
    """Hand-built dataset with a KNOWN, large, deliberate between-site
    difference on 'score_a' and NO real difference (just noise) on
    'score_b' -- so ANOVA correctness can be checked against a known
    ground truth rather than trusting the function's own output.
    """
    rows = []
    # Site X: score_a clustered around 4.5-5; score_b noisy around 3.0
    for a, b in zip([4.5, 4.7, 4.6, 4.8, 4.9, 4.4], [2.8, 3.1, 2.9, 3.2, 3.0, 2.95]):
        rows.append({"site": "SiteX", "score_a": a, "score_b": b})
    # Site Y: score_a clustered around 1.0-1.5 (clearly different from X);
    # score_b drawn from the same noisy ~3.0 distribution as SiteX (no real
    # site difference on this variable, just sampling noise)
    for a, b in zip([1.0, 1.2, 1.1, 1.3, 1.4, 1.0], [3.05, 2.9, 3.1, 2.85, 3.15, 3.0]):
        rows.append({"site": "SiteY", "score_a": a, "score_b": b})
    return pd.DataFrame(rows)


class TestCompareSitesAnova:
    def test_detects_a_known_large_difference(self):
        df = _controlled_df()
        result = compare_sites_anova(df, "score_a")
        assert result["significant_at_05"] is True
        assert result["p_value"] < 0.05
        assert result["f_statistic"] > 0

    def test_does_not_flag_a_known_non_difference(self):
        df = _controlled_df()
        result = compare_sites_anova(df, "score_b")
        assert result["significant_at_05"] is False
        assert result["p_value"] >= 0.05

    def test_site_summary_contains_every_site(self):
        df = _controlled_df()
        result = compare_sites_anova(df, "score_a")
        assert set(result["site_summary"].keys()) == {"SiteX", "SiteY"}

    def test_raises_with_fewer_than_two_sites(self):
        df = pd.DataFrame({"site": ["SiteX"] * 4, "score_a": [1, 2, 3, 4]})
        with pytest.raises(ValueError):
            compare_sites_anova(df, "score_a")

    def test_matches_scipy_f_oneway_directly(self):
        from scipy import stats
        df = _controlled_df()
        result = compare_sites_anova(df, "score_a")
        x = df.loc[df["site"] == "SiteX", "score_a"]
        y = df.loc[df["site"] == "SiteY", "score_a"]
        f_expected, p_expected = stats.f_oneway(x, y)
        assert result["f_statistic"] == pytest.approx(f_expected)
        assert result["p_value"] == pytest.approx(p_expected)


class TestCompareAllConstructs:
    """compare_all_constructs takes BARE construct names and internally
    looks up f"{name}_score" -- so the controlled fixture's score columns
    need to be renamed to match that f"{name}_score" convention.
    """

    def _renamed_df(self):
        df = _controlled_df()
        return df.rename(columns={"score_a": "a_score", "score_b": "b_score"})

    def test_returns_one_row_per_construct_name(self):
        df = self._renamed_df()
        result = compare_all_constructs(df, ["a", "b"])
        assert len(result) == 2
        assert set(result["construct"]) == {"a", "b"}

    def test_flags_match_individual_calls(self):
        df = self._renamed_df()
        result = compare_all_constructs(df, ["a", "b"])
        row_a = result.loc[result["construct"] == "a"].iloc[0]
        row_b = result.loc[result["construct"] == "b"].iloc[0]
        assert bool(row_a["significant_at_05"]) is True
        assert bool(row_b["significant_at_05"]) is False


class TestPairwiseSiteGaps:
    def test_identifies_a_large_known_gap(self):
        df = _controlled_df()
        gaps = pairwise_site_gaps(df, "score_a", min_gap=0.4)
        assert len(gaps) == 1
        row = gaps.iloc[0]
        assert {row["site_a"], row["site_b"]} == {"SiteX", "SiteY"}
        assert row["gap"] > 3.0  # true means are ~4.65 vs ~1.17

    def test_excludes_gaps_below_threshold(self):
        df = _controlled_df()
        gaps = pairwise_site_gaps(df, "score_b", min_gap=0.4)
        assert len(gaps) == 0

    def test_higher_threshold_can_exclude_a_real_but_smaller_gap(self):
        # score_b sites differ by ~0.05 -- with a very low threshold this
        # tiny gap should be caught; with the default it should not.
        df = _controlled_df()
        tiny_threshold_gaps = pairwise_site_gaps(df, "score_b", min_gap=0.01)
        assert len(tiny_threshold_gaps) == 1
        default_gaps = pairwise_site_gaps(df, "score_b", min_gap=0.4)
        assert len(default_gaps) == 0

    def test_gap_value_matches_manual_mean_difference(self):
        df = _controlled_df()
        gaps = pairwise_site_gaps(df, "score_a", min_gap=0.4)
        manual_gap = abs(df.loc[df["site"] == "SiteX", "score_a"].mean() - df.loc[df["site"] == "SiteY", "score_a"].mean())
        assert gaps.iloc[0]["gap"] == pytest.approx(round(manual_gap, 3))


class TestMeasurementInvarianceNote:
    def test_note_is_a_nonempty_string_naming_the_omitted_technique(self):
        assert isinstance(MEASUREMENT_INVARIANCE_NOTE, str)
        assert len(MEASUREMENT_INVARIANCE_NOTE) > 50
        assert "measurement invariance" in MEASUREMENT_INVARIANCE_NOTE.lower()


class TestOnRealGeneratedData:
    """End-to-end sanity check using the actual simulator + scoring
    pipeline (not just hand-built controlled data): the real injected
    Amberg manager_support gap should be detected as both statistically
    significant and practically large.
    """

    def test_amberg_manager_support_gap_is_detected_end_to_end(self):
        from src.collect_data import generate_responses
        from src.quality_checks import clean_responses
        from src.scoring import score_all_constructs

        df = generate_responses(seed=42)
        cleaned = clean_responses(df)
        scored = score_all_constructs(cleaned)

        anova_result = compare_sites_anova(scored, "manager_support_score")
        assert anova_result["significant_at_05"] is True

        gaps = pairwise_site_gaps(scored, "manager_support_score", min_gap=0.4)
        amberg_gaps = gaps[(gaps["site_a"] == "Amberg") | (gaps["site_b"] == "Amberg")]
        assert len(amberg_gaps) > 0
