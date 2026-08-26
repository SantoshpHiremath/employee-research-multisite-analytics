"""
End-to-end demo: simulates multi-site survey collection, cleans the
responses (straightliner + low-completion removal), scores the three
constructs, reports internal-consistency reliability (Cronbach's alpha),
and runs the cross-site comparability checks (ANOVA + practical-gap
flagging) -- printing an honest summary, including the reliability bands
and the explicit measurement-invariance disclosure.
"""
import os

from src.collect_data import generate_responses
from src.quality_checks import (
    flag_straightliners, item_missingness_rate, respondent_completion_rate,
    clean_responses,
)
from src.scoring import score_all_constructs, construct_reliability_report
from src.comparability import compare_all_constructs, pairwise_site_gaps, MEASUREMENT_INVARIANCE_NOTE
from src.instrument import CONSTRUCTS


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("1. DATA COLLECTION (synthetic, multi-site)")
    print("=" * 70)
    raw = generate_responses(seed=42)
    raw.to_csv("data/survey_responses_raw.csv", index=False)
    print(f"Collected {len(raw)} responses across {raw['site'].nunique()} sites.")
    print(raw["site"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("2. DATA QUALITY CHECKS")
    print("=" * 70)
    straightliners = flag_straightliners(raw)
    completion = respondent_completion_rate(raw)
    print(f"Likely straightliners flagged: {straightliners.sum()} / {len(raw)} ({straightliners.mean():.1%})")
    print(f"Mean respondent completion rate: {completion.mean():.1%}")
    print("Per-item missingness rate:")
    print(item_missingness_rate(raw).round(3).to_string())

    cleaned = clean_responses(raw, min_completion_rate=0.75)
    cleaned.to_csv("data/survey_responses_cleaned.csv", index=False)
    print(f"\nRemaining after cleaning: {len(cleaned)} / {len(raw)} respondents "
          f"({len(raw) - len(cleaned)} removed).")

    print("\n" + "=" * 70)
    print("3. CONSTRUCT SCORING + RELIABILITY (Cronbach's alpha)")
    print("=" * 70)
    scored = score_all_constructs(cleaned)
    reliability = construct_reliability_report(cleaned)
    print(reliability.to_string(index=False))

    print("\n" + "=" * 70)
    print("4. CROSS-SITE COMPARABILITY")
    print("=" * 70)
    construct_names = [c.name for c in CONSTRUCTS]
    anova_results = compare_all_constructs(scored, construct_names)
    print(anova_results.to_string(index=False))

    print("\nPractically large site-to-site gaps (>= 0.4 points on the 1-5 scale):")
    any_gaps = False
    for name in construct_names:
        gaps = pairwise_site_gaps(scored, f"{name}_score", min_gap=0.4)
        if len(gaps) > 0:
            any_gaps = True
            print(f"\n  {name}:")
            print("  " + gaps.to_string(index=False).replace("\n", "\n  "))
    if not any_gaps:
        print("  (none)")

    print("\n" + "-" * 70)
    print("Measurement invariance disclosure:")
    print(MEASUREMENT_INVARIANCE_NOTE)

    print("\n" + "=" * 70)
    print("5. OVERALL SITE MEANS BY CONSTRUCT")
    print("=" * 70)
    summary = scored.groupby("site")[[f"{c.name}_score" for c in CONSTRUCTS]].mean().round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
