import pytest

from src.instrument import (
    SURVEY_ITEMS, CONSTRUCTS, get_item, get_construct, reverse_code,
    LIKERT_MIN, LIKERT_MAX,
)


class TestSurveyDesign:
    def test_every_construct_has_at_least_two_items(self):
        """Cronbach's alpha requires at least 2 items -- a design
        constraint, not just a nice-to-have.
        """
        for construct in CONSTRUCTS:
            assert len(construct.item_ids) >= 2

    def test_every_construct_has_at_least_one_reverse_coded_item(self):
        """Regression test for a real survey-design requirement: every
        construct needs a reverse-coded item to make straightlining
        detectable (see quality_checks.py).
        """
        for construct in CONSTRUCTS:
            items = [get_item(iid) for iid in construct.item_ids]
            assert any(item.reverse_coded for item in items)

    def test_every_item_belongs_to_a_defined_construct(self):
        construct_names = {c.name for c in CONSTRUCTS}
        for item in SURVEY_ITEMS:
            assert item.construct in construct_names

    def test_every_construct_item_id_resolves_to_a_real_item(self):
        for construct in CONSTRUCTS:
            for item_id in construct.item_ids:
                get_item(item_id)  # raises if not found


class TestReverseCode:
    def test_reverse_code_is_involutive(self):
        """Applying reverse_code twice returns the original value."""
        for v in range(LIKERT_MIN, LIKERT_MAX + 1):
            assert reverse_code(reverse_code(v)) == v

    def test_reverse_code_maps_extremes_correctly(self):
        assert reverse_code(LIKERT_MIN) == LIKERT_MAX
        assert reverse_code(LIKERT_MAX) == LIKERT_MIN

    def test_reverse_code_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            reverse_code(0)
        with pytest.raises(ValueError):
            reverse_code(6)


class TestGetters:
    def test_get_item_raises_for_unknown_id(self):
        with pytest.raises(ValueError):
            get_item("NOT_REAL")

    def test_get_construct_raises_for_unknown_name(self):
        with pytest.raises(ValueError):
            get_construct("not_a_construct")
