from app.agents.hypothesis_auditor_agent import deterministic_hypothesis_audit
from app.tools.hypothesis_scientific_guard import guard_hypothesis_analysis


NO_DIRECT_CONTEXT = (
    "LiteratureAgent grounding unavailable. Directly relevant evidence: 0. "
    "Treat the proposal as unverified."
)


def test_live_ed_binding_and_pl_overclaims_are_calibrated() -> None:
    report = """
    The ethylenediamine dication (ED2+) has higher charge density and tighter binding than
    monovalent spacers. It should result in a measurable increase in the lattice parameter,
    indicative of tighter binding. The unit cell volume is predicted to decrease, reflecting
    enhanced electrostatic interactions. A lower formation energy would support tighter binding.
    The defect density, as measured by photoluminescence spectroscopy, would be reduced.
    Photoluminescence (PL) spectroscopy will be utilized to evaluate the defect density by
    comparing PL intensity and lifetime. SEM and TEM will visualize morphology, providing direct
    evidence of the impact of ED2+ on the crystal lattice.
    """

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "eda2+ denotes doubly protonated ethylenediamine" in lower
    assert "cannot isolate valence alone" in lower
    assert "tighter binding" not in lower
    assert "higher charge density" not in lower
    assert "increase in the lattice parameter" not in lower
    assert "defect density, as measured by photoluminescence" not in lower
    assert "will be utilized to evaluate the defect density" not in lower
    assert "providing direct evidence" not in lower
    assert not deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT).findings


def test_auditor_detects_lattice_and_formation_energy_overreach() -> None:
    report = (
        "The lattice parameter will increase while the unit-cell volume will decrease. "
        "A lower formation energy proves tighter binding."
    )

    issues = " ".join(
        finding.issue for finding in deterministic_hypothesis_audit(report, NO_DIRECT_CONTEXT).findings
    ).lower()

    assert "increased lattice parameter and a decreased unit-cell volume" in issues
    assert "formation energy is used as though it directly establishes" in issues


def test_unsupported_directional_pce_language_is_secondary() -> None:
    report = """
    No direct evidence supports the ion migration mechanism. The power conversion efficiency
    (PCE) of treated cells will increase measurably relative to controls. The key parameter is the
    PCE value, which should be higher in treated cells, confirming the mechanism. An increase in
    PCE would support the proposed benefits.
    """

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "will increase measurably" not in lower
    assert "should be higher" not in lower
    assert "increase in pce would support" not in lower
    assert "secondary device-level" in lower
    assert "without a prespecified direction" in lower
    assert not deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT).findings


def test_live_tof_tpv_pce_novelty_and_formula_variants_are_calibrated() -> None:
    report = """
    No directly relevant evidence was retrieved. ToF-SIMS will be used to measure the ion migration
    rates. The key parameter here is the rate of ion displacement over time, which will provide
    direct evidence of suppression. The PCE is expected to be measurably higher. TPV and TPC decay
    times and magnitudes of the transient signals are indicative of charge carrier lifetimes and
    mobilities. Use current-voltage (I-V) curve analysis. This application appears novel, as direct
    supporting evidence is currently lacking. The proposed tests are well-designed to measure the
    mechanism. Continue experiments in FAPB3.
    """

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "will be used to measure the ion migration rates" not in lower
    assert "will provide direct evidence" not in lower
    assert "expected to be measurably higher" not in lower
    assert "indicative of charge carrier lifetimes and mobilities" not in lower
    assert "current-voltage (i-v)" not in lower
    assert "appears novel" not in lower
    assert "well-designed" not in lower
    assert "fapb3" not in lower
    assert "fapbi3" in lower
    assert "do not independently measure intrinsic carrier lifetime or mobility" in lower
    assert "novelty cannot be inferred" in lower
    assert not deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT).findings


def test_live_eis_tpv_thermal_pea_and_dsc_boundary_variants_are_calibrated() -> None:
    report = """
    No directly relevant evidence was retrieved. Similarly, the conjugated pi-system of PEA may facilitate charge
    transport through hopping or tunneling processes. EIS will be employed to measure ion migration
    rates. The key parameter here is the resistance associated with ion migration, which should be
    higher. This test is crucial as it directly measures the impact on ion dynamics. Transient
    Photovoltage Measurements: These measurements will assess the charge transport efficiency. By
    comparing the decay times of photogenerated carriers, we can determine if the PEA interlayer
    affects charge carrier mobility. The prediction that the interlayer will maintain its structural
    integrity and functionality up to 80 C is testable. The hypothesis retains a degree of novelty as
    an unverified scientific exploration and introduces a novel combination. The proposed tests are
    well-suited to evaluate the hypothesis with promising potential. DSC does not directly assign a
    spacer-lattice binding energy.
    """

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "hopping or tunneling" not in lower
    assert "measure ion migration rates" not in lower
    assert "directly measures the impact" not in lower
    assert "assess the charge transport efficiency" not in lower
    assert "affects charge carrier mobility" not in lower
    assert "maintain its structural integrity and functionality up to 80 c" not in lower
    assert "retains a degree of novelty" not in lower
    assert "introduces a novel combination" not in lower
    assert "well-suited" not in lower
    assert "promising potential" not in lower
    assert "does not directly assign a spacer-lattice binding energy" in lower
    assert "do not independently measure intrinsic carrier lifetime or mobility" in lower
    assert not deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT).findings


def test_pea_live_report_methods_controls_and_praise_are_calibrated() -> None:
    report = """
    No directly relevant evidence was retrieved. PEA's flexible alkyl chain forms strong hydrogen
    bonds and suppresses ion migration. Controls without the PEA interlayer will be compared.
    An increase in the power conversion efficiency (PCE) is predicted. The treated samples are
    expected to show improved crystallinity and fewer defects using XRD and SEM. Utilize
    time-resolved X-ray diffraction (TR-XRD) and secondary ion mass spectrometry (SIMS) to monitor
    the movement of ions across the interface. These techniques will provide direct evidence of ion
    migration rates. The experimental tests outlined are robust and well-suited to validate or
    falsify the hypothesis. The mechanism represents a novel approach and introduces a fresh
    perspective.
    """

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "flexible alkyl chain" not in lower
    assert "strong hydrogen bonds" not in lower
    assert "direct evidence of ion migration rates" not in lower
    assert "increase in the power conversion efficiency" not in lower
    assert "improved crystallinity and fewer defects" not in lower
    assert "robust and well-suited" not in lower
    assert "represent a novel approach" not in lower
    assert "introduces a fresh perspective" not in lower
    assert "cannot isolate the pea mechanism" in lower
    assert "neither alone directly measures an ion-migration rate" in lower
    assert "without a prespecified direction" in lower
    assert not deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT).findings
