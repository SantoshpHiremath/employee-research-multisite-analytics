import pandas as pd

from src.collect_data import generate_responses, SITES, SITE_TRUE_MEANS
from src.instrument import SURVEY_ITEMS, LIKERT_MIN, LIKERT_MAX


class TestGenerateResponses:
    def test_same_seed_is_deterministic(self):
        df1 = generate_responses(seed=1)
        df2 = generate_responses(seed=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self):
        df1 = generate_responses(seed=1)
        df2 = generate_responses(seed=2)
        assert not df1["PS1"].equals(df2["PS1"])

    def test_covers_every_configured_site(self):
        df = generate_responses(seed=1)
        assert set(df["site"]) == set(SITES)

    def test_all_responses_within_likert_range_or_missing(self):
        df = generate_responses(seed=1)
        item_ids = [item.item_id for item in SURVEY_ITEMS]
        for col in item_ids:
            valid = df[col].dropna()
            assert (valid >= LIKERT_MIN).all()
            assert (valid <= LIKERT_MAX).all()

    def test_has_some_missing_values(self):
        """Confirms the injected item-level missingness actually shows up
        -- not testing for an exact rate (that would be a flaky test
        given randomness), just that it's a real, present phenomenon in
        the generated data.
        """
        df = generate_responses(seed=1)
        item_ids = [item.item_id for item in SURVEY_ITEMS]
        assert df[item_ids].isna().sum().sum() > 0


class TestSiteMeanRecovery:
    """The most important test in this file: does the pipeline (as a
    whole, from raw generation through scoring) actually recover the
    TRUE site-level differences that were injected into the simulator?
    If not, no amount of individual-function unit testing means the
    project's core claim ("multi-site data reveals real between-site
    differences") is trustworthy.
    """

    def test_amberg_manager_support_is_recovered_as_lower_than_other_sites(self):
        from src.quality_checks import clean_responses
        from src.scoring import score_all_constructs

        df = generate_responses(seed=42)
        cleaned = clean_responses(df)
        scored = score_all_constructs(cleaned)

        site_means = scored.groupby("site")["manager_support_score"].mean()
        # Amberg's true mean (2.7) is deliberately the lowest configured
        assert site_means["Amberg"] == min(site_means)
        # and the gap to the next-lowest site should be substantial, not
        # noise-level
        other_sites_min = site_means.drop("Amberg").min()
        assert other_sites_min - site_means["Amberg"] > 0.3
