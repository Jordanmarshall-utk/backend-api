"""Experiment session helpers."""

from app.services.experiment_service import (
    _blocked_experiment_draft,
    _experiment_automation_blockers,
    patch_manual_inputs,
)


class _Mem:
    def __init__(self) -> None:
        self._data: dict = {}

    def get_var(self, key: str):
        return self._data.get(key)

    def set_var(self, key: str, value) -> None:
        self._data[key] = value


def test_patch_manual_inputs_passes_memory():
    mem = _Mem()
    patch_manual_inputs(mem, {"manual_hypothesis": "H1", "manual_thoughts": "T1"})
    assert mem.get_var("manual_hypothesis") == "H1"
    assert mem.get_var("manual_thoughts") == "T1"


def test_automation_gate_blocks_empty_constraints_and_invented_equipment():
    constraints = {
        "techniques": [],
        "equipment": [],
        "liquid_handling": {"materials": [], "instruments": []},
    }
    blockers = _experiment_automation_blockers(
        constraints,
        "Use a Tecan to mix PEA and FAPbI3 and form an interlayer.",
    )
    joined = " ".join(blockers)
    assert "material" in joined.lower()
    assert "liquid-handling" in joined.lower()
    assert "Tecan" in joined
    assert "solution mixing is not an interface" in joined


def test_automation_gate_accepts_a_configured_controlled_interface_plan():
    constraints = {
        "techniques": ["XRD", "impedance spectroscopy"],
        "equipment": ["spin coater"],
        "liquid_handling": {
            "materials": ["PEA stock", "FAPbI3 precursor", "solvent"],
            "instruments": ["manual pipettes"],
        },
    }
    plan = (
        "Spin-coat films on randomized substrates in triplicate (n=3). Include a no-interlayer "
        "control, a sham-processed control, a thickness-matched control, and an alternative spacer. "
        "Use XRD for phase verification and impedance spectroscopy under fixed conditions."
    )
    assert _experiment_automation_blockers(constraints, plan) == []


def test_blocked_draft_does_not_expose_rejected_worklist_or_volumes():
    draft = _blocked_experiment_draft(
        "How can an interlayer suppress ion migration?",
        ["Configure materials.", "Configure characterization."],
    )
    assert "How can an interlayer suppress ion migration?" in draft
    assert "Withheld until the release gate passes" in draft
    assert "Configure materials" in draft
    assert "A1," not in draft
    assert "uL=" not in draft
