"""
Cross-site comparability checks -- the real methodological core of this
project. Pooling multi-site survey data and reporting one overall number
is only valid if scores are actually comparable across sites; if one
site's respondents interpret the scale differently, or genuinely
experience different conditions, treating "one big pooled sample" as
representative can hide real, important between-site differences (or
worse, manufacture a fake overall trend from mixing populations that
shouldn't be combined without disaggregation).

This module checks comparability at two levels of rigor:

1. A real, standard one-way ANOVA per construct testing whether sites
   differ significantly -- the correct question to ask BEFORE pooling
   ("do these sites actually differ?"), not skipped in favor of jumping
   straight to an overall mean.
2. A scope note on full measurement invariance testing (the
   gold-standard multi-group CFA technique for checking whether a scale
   means the same thing across groups), identified as the natural next
   step beyond the ANOVA and practical-gap checks implemented here.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats


def compare_sites_anova(df_scored: pd.DataFrame, construct_score_col: str) -> dict:
    """One-way ANOVA testing whether mean construct score differs across
    sites. Returns the F-statistic, p-value, and per-site means/counts,
    so both statistical significance AND practical magnitude are
    reported together (a significant p-value alone doesn't tell you
    whether the difference is large enough to matter operationally).
    """
    sites = df_scored["site"].unique()
    groups = [
        df_scored.loc[df_scored["site"] == site, construct_score_col].dropna()
        for site in sites
    ]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError("Need at least 2 sites with data to compare")

    f_stat, p_value = stats.f_oneway(*groups)

    site_summary = (
        df_scored.groupby("site")[construct_score_col]
        .agg(["mean", "std", "count"])
        .round(3)
        .to_dict(orient="index")
    )

    return {
        "construct": construct_score_col,
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "site_summary": site_summary,
    }


def compare_all_constructs(df_scored: pd.DataFrame, construct_names: list) -> pd.DataFrame:
    rows = []
    for name in construct_names:
        result = compare_sites_anova(df_scored, f"{name}_score")
        rows.append({
            "construct": name,
            "f_statistic": round(result["f_statistic"], 3),
            "p_value": round(result["p_value"], 4),
            "significant_at_05": result["significant_at_05"],
        })
    return pd.DataFrame(rows)


def pairwise_site_gaps(df_scored: pd.DataFrame, construct_score_col: str, min_gap: float = 0.4) -> pd.DataFrame:
    """Flags site pairs whose mean construct score differs by at least
    `min_gap` points on the 1-5 scale -- a practically meaningful
    threshold check ALONGSIDE the statistical-significance test, since a
    tiny but statistically-significant gap (common with large samples)
    isn't necessarily operationally important, and a large gap in a
    smaller/noisier sample might not reach significance but is still
    worth a human looking at.
    """
    site_means = df_scored.groupby("site")[construct_score_col].mean()
    rows = []
    sites = list(site_means.index)
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            site_a, site_b = sites[i], sites[j]
            gap = abs(site_means[site_a] - site_means[site_b])
            if gap >= min_gap:
                rows.append({
                    "site_a": site_a, "site_b": site_b,
                    "mean_a": round(site_means[site_a], 3),
                    "mean_b": round(site_means[site_b], 3),
                    "gap": round(gap, 3),
                })
    return pd.DataFrame(rows)


MEASUREMENT_INVARIANCE_NOTE = (
    "This project checks whether site MEANS differ (one-way ANOVA) and "
    "flags practically large gaps. Full measurement invariance testing "
    "(configural/metric/scalar invariance via multi-group "
    "confirmatory factor analysis) is the natural next step: the "
    "gold-standard check for whether a survey scale means the same thing "
    "to respondents across groups before comparing their means at all. "
    "It requires a larger per-site sample and CFA tooling, and is "
    "planned as an extension."
)
