"""
Research instrument design: a structured employee-experience survey with
defined, named constructs, each measured by multiple 5-point Likert items
(not a single ad-hoc question per topic) -- the standard, defensible way
to design a survey construct, since a single item is noisy and can't be
checked for internal consistency, while multiple items measuring the same
underlying construct can be (via Cronbach's alpha, see analysis.py).

Built for a research-design + multi-site data collection use case: this
module is the research design, made concrete and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SurveyItem:
    item_id: str
    construct: str
    text: str
    reverse_coded: bool = False  # True if a HIGH raw response means LOW construct level


@dataclass
class Construct:
    name: str
    description: str
    item_ids: list


# Three named constructs, each with 4 items (a real, standard survey-design
# minimum for checking internal consistency -- a single item can't be
# checked this way). One reverse-coded item per construct is included
# deliberately: a well-known survey-design technique to catch respondents
# who straight-line (select the same answer for every item without
# reading them), since a straight-liner will contradict themselves on a
# reverse-coded item.
SURVEY_ITEMS = [
    # Construct 1: Psychological Safety (adapted from the well-established
    # Edmondson (1999) psychological safety construct -- publicly
    # documented survey research, not invented from scratch)
    SurveyItem("PS1", "psychological_safety", "I feel safe raising concerns or mistakes with my team."),
    SurveyItem("PS2", "psychological_safety", "My team values my input, even when it differs from the majority."),
    SurveyItem("PS3", "psychological_safety", "It is difficult to ask others on my team for help.", reverse_coded=True),
    SurveyItem("PS4", "psychological_safety", "I can take a risk on my team without being penalized."),

    # Construct 2: Workload Sustainability
    SurveyItem("WS1", "workload_sustainability", "My current workload is manageable within normal working hours."),
    SurveyItem("WS2", "workload_sustainability", "I regularly feel I do not have enough time to complete my tasks well.", reverse_coded=True),
    SurveyItem("WS3", "workload_sustainability", "I can maintain my current pace of work without burning out."),
    SurveyItem("WS4", "workload_sustainability", "My workload has been reasonable over the past month."),

    # Construct 3: Manager Support
    SurveyItem("MS1", "manager_support", "My manager provides clear guidance when I need it."),
    SurveyItem("MS2", "manager_support", "My manager recognizes my contributions."),
    SurveyItem("MS3", "manager_support", "I rarely receive useful feedback from my manager.", reverse_coded=True),
    SurveyItem("MS4", "manager_support", "My manager supports my professional development."),
]

CONSTRUCTS = [
    Construct(
        "psychological_safety",
        "Whether employees feel safe raising concerns, mistakes, or dissenting views.",
        ["PS1", "PS2", "PS3", "PS4"],
    ),
    Construct(
        "workload_sustainability",
        "Whether employees' current workload is sustainable without burnout risk.",
        ["WS1", "WS2", "WS3", "WS4"],
    ),
    Construct(
        "manager_support",
        "Whether employees feel supported and developed by their direct manager.",
        ["MS1", "MS2", "MS3", "MS4"],
    ),
]

LIKERT_MIN, LIKERT_MAX = 1, 5

SITES = ["Munich", "Regensburg", "Erlangen", "Amberg"]


def get_item(item_id: str) -> SurveyItem:
    matches = [i for i in SURVEY_ITEMS if i.item_id == item_id]
    if not matches:
        raise ValueError(f"Unknown item_id: {item_id}")
    return matches[0]


def get_construct(name: str) -> Construct:
    matches = [c for c in CONSTRUCTS if c.name == name]
    if not matches:
        raise ValueError(f"Unknown construct: {name}")
    return matches[0]


def reverse_code(raw_response: int) -> int:
    """Standard Likert reverse-coding: on a 1-5 scale, response r maps to
    (LIKERT_MAX + LIKERT_MIN - r), so a '5' becomes a '1' and vice versa.
    """
    if not (LIKERT_MIN <= raw_response <= LIKERT_MAX):
        raise ValueError(f"Response {raw_response} outside Likert range [{LIKERT_MIN},{LIKERT_MAX}]")
    return LIKERT_MAX + LIKERT_MIN - raw_response
