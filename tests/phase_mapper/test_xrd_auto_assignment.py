import numpy as np
import pandas as pd

from app.services.phase_mapper.xrd_auto_assignment import (
    merge_proposals_into_draft_labels,
    propose_reference_labels,
    score_reference_patterns,
)


def _peaks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"library": "Rb", "sample_id": "A1", "well": "A1", "source_file": "A1.xrdml", "peak_id": 1, "two_theta": 10.02, "area": 100.0, "height": 100.0, "fwhm": 0.1, "status": "auto"},
            {"library": "Rb", "sample_id": "A1", "well": "A1", "source_file": "A1.xrdml", "peak_id": 2, "two_theta": 20.03, "area": 80.0, "height": 80.0, "fwhm": 0.1, "status": "auto"},
            {"library": "Rb", "sample_id": "A1", "well": "A1", "source_file": "A1.xrdml", "peak_id": 3, "two_theta": 30.04, "area": 60.0, "height": 60.0, "fwhm": 0.1, "status": "auto"},
        ]
    )


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"phase": "RbPbI3", "reference_two_theta": 10.0, "relative_intensity": 100.0, "tolerance": 0.25, "selected": True, "visible": True},
            {"phase": "RbPbI3", "reference_two_theta": 20.0, "relative_intensity": 80.0, "tolerance": 0.25, "selected": True, "visible": True},
            {"phase": "RbPbI3", "reference_two_theta": 30.0, "relative_intensity": 60.0, "tolerance": 0.25, "selected": True, "visible": True},
        ]
    )


def test_reference_pattern_and_composition_propose_named_phase():
    composition = pd.DataFrame([{"library": "Rb", "well": "A1", "Rb_fraction": 0.25, "DMA_fraction": 0.0, "Cs_fraction": 0.0}])
    scores = score_reference_patterns(_peaks(), _references(), composition=composition)
    proposals = propose_reference_labels(scores)

    assert scores.iloc[0]["composition_compatible"]
    assert scores.iloc[0]["matched_reference_peaks"] == 3
    assert proposals.iloc[0]["xrd_RbPbI3_present"] == 1
    assert proposals.iloc[0]["xrd_secondary_phase_present"] == 1
    assert proposals.iloc[0]["xrd_mixed_phase"] == 1
    assert proposals.iloc[0]["xrd_clean_3d_perovskite"] == 0


def test_reference_pattern_rejects_composition_incompatible_phase():
    composition = pd.DataFrame([{"library": "Rb", "well": "A1", "Rb_fraction": 0.0, "DMA_fraction": 0.0, "Cs_fraction": 0.0}])
    scores = score_reference_patterns(_peaks(), _references(), composition=composition)
    proposals = propose_reference_labels(scores)

    assert not bool(scores.iloc[0]["composition_compatible"])
    assert "xrd_RbPbI3_present" not in proposals.columns or pd.isna(proposals.iloc[0].get("xrd_RbPbI3_present", np.nan))
    assert proposals.iloc[0]["xrd_auto_assignment_status"] == "no_confident_reference_label"


def test_reference_pattern_requires_a_diagnostic_nonshared_peak():
    references = _references()
    shared = references.copy()
    shared["phase"] = "Other phase"
    scores = score_reference_patterns(
        _peaks(),
        pd.concat([references, shared], ignore_index=True),
        composition=pd.DataFrame([{"library": "Rb", "well": "A1", "Rb_fraction": 0.25}]),
    )
    proposals = propose_reference_labels(scores)

    rb_score = scores[scores["phase"].eq("RbPbI3")].iloc[0]
    assert rb_score["unique_matched_reference_peaks"] == 0
    assert proposals.iloc[0]["xrd_auto_assignment_status"] == "no_confident_reference_label"


def test_reference_proposal_overrides_conflicting_automatic_rule_draft():
    draft = pd.DataFrame(
        [
            {
                "xrd_library": "Rb",
                "well": "A1",
                "xrd_clean_3d_perovskite": 1,
                "xrd_mixed_phase": 0,
                "xrd_assignment_confidence": "low",
                "xrd_notes": "Rule draft from raw XRD",
            }
        ]
    )
    proposals = pd.DataFrame(
        [
            {
                "library": "Rb",
                "well": "A1",
                "xrd_RbPbI3_present": 1.0,
                "xrd_secondary_phase_present": 1.0,
                "xrd_mixed_phase": 1.0,
                "xrd_clean_3d_perovskite": 0.0,
                "xrd_auto_assignment_status": "proposed_needs_review",
                "xrd_auto_assignment_confidence": "medium",
                "xrd_auto_assigned_phases": "RbPbI3",
                "xrd_auto_ambiguous_phases": "",
                "xrd_auto_assignment_note": "Automatic reference-pattern and composition proposal; manual XRD review remains authoritative.",
            }
        ]
    )

    merged = merge_proposals_into_draft_labels(draft, proposals).iloc[0]

    assert merged["xrd_clean_3d_perovskite"] == 0
    assert merged["xrd_mixed_phase"] == 1
    assert merged["xrd_RbPbI3_present"] == 1
    assert merged["xrd_label_source"] == "automatic_reference_pattern_and_composition_proposal"
    assert merged["xrd_final_label_basis"] == "reference_pattern_override_needs_review"
