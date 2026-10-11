import pytest

from app.services.hypothesis_chat import _resolve_choice


class ChoiceMemory:
    def __init__(self, stage="refine"):
        self.values = {"stage": stage}
        self.components = {
            "first_thought_1": "First branch",
            "second_thought_1": "Second branch",
            "third_thought_1": "Third branch",
            "next_step_option_1": "Next one",
            "next_step_option_2": "Next two",
            "next_step_option_3": "Next three",
        }

    def get_var(self, key, default=None):
        return self.values.get(key, default)

    def view_component(self, key):
        return self.components.get(key)


def test_resolves_numeric_and_exact_refine_choices():
    memory = ChoiceMemory()
    assert _resolve_choice(memory, "2") == "2"
    assert _resolve_choice(memory, "Second branch") == "2"


def test_resolves_exact_continuation_choice():
    memory = ChoiceMemory(stage="hypothesis")
    assert _resolve_choice(memory, "Next three") == "3"


def test_rejects_unknown_choice_instead_of_selecting_third():
    with pytest.raises(ValueError, match="exactly match"):
        _resolve_choice(ChoiceMemory(), "Unknown branch")
