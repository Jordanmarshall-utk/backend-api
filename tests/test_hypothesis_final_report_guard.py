from app.agents.hypothesis_auditor_agent import deterministic_hypothesis_audit
from app.tools.hypothesis_scientific_guard import (
    guard_hypothesis_analysis,
    guard_hypothesis_synthesis,
)


NO_DIRECT_CONTEXT = (
    "No directly relevant evidence was retrieved. "
    "The EDA2+ versus PEA+ mechanism remains an unverified proposal."
)


def test_synthesis_removes_non_direct_anchors_and_calibrates_methods() -> None:
    report = (
        "No direct empirical evidence supports this hypothesis. "
        "[KG2] DOI: 10.3390/nano15110798 validates the proposal. "
        "EDA2+ has coordination capabilities, stronger electrostatic interactions, "
        "and a denser network of bonds than PEA+. "
        "ToF-SIMS will directly measure ion migration rates. "
        "TRMC will measure carrier mobility. "
        "I-V measurements will measure device performance."
    )

    guarded = guard_hypothesis_synthesis(report)
    lower = guarded.lower()

    assert "[kg2]" not in lower
    assert "10.3390/nano15110798" not in lower
    assert "coordination capabilities" not in lower
    assert "denser network of bonds" not in lower
    assert "cannot isolate valence alone" in lower
    assert "tof-sims will directly measure ion migration rates" not in lower
    assert "trmc will measure carrier mobility" not in lower
    assert "current-density-voltage (j-v)" in lower
    assert "stabilized maximum-power-point" in lower
    assert "eqe consistency" in lower


def test_analysis_guard_is_idempotent_and_removes_certainty_language() -> None:
    report = (
        "The proposal is highly testable because all of the methods are well-defined "
        "and directly measure the relevant outcomes. ToF-SIMS will quantify migration rates."
    )

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    guarded_twice = guard_hypothesis_analysis(guarded, NO_DIRECT_CONTEXT)
    lower = guarded_twice.lower()

    assert guarded_twice.count("**Deterministic scientific audit**") == 1
    assert "highly testable" not in lower
    assert "tof-sims will quantify migration rates" not in lower
    assert "conditionally testable" in lower or "speculative but testable" in lower


def test_measurement_acronyms_are_not_treated_as_unknown_chemicals() -> None:
    audit = deterministic_hypothesis_audit(
        "TRMC, I-V, J-V, MPP, EQE, RH, and ToF-SIMS are measurement terms.",
        "",
    )

    assert all("chemical identity" not in item.message.lower() for item in audit.findings)


def test_auditor_catches_valence_eis_pl_and_network_overclaims() -> None:
    report = (
        "No direct evidence supports the mechanism. The divalent nature of EDA2+ creates a "
        "more robust barrier than a monovalent spacer by forming a denser and more stable network. "
        "Use electrochemical impedance spectroscopy (EIS) to measure ion conductivity; these "
        "measurements directly assess ion movement. Use photoluminescence (PL) to assess defect "
        "density, with a higher signal indicating fewer defects."
    )

    audit = deterministic_hypothesis_audit(report, "")
    categories = [item.category for item in audit.findings]

    assert "design_confounding" in categories
    assert "mechanism_calibration" in categories
    assert categories.count("measurement_validity") >= 2


def test_guard_resolves_valence_eis_pl_and_network_contracts() -> None:
    report = (
        "No direct evidence supports the mechanism. The divalent nature of EDA2+ creates a "
        "more robust barrier than a monovalent spacer by forming a denser and more stable network. "
        "Use electrochemical impedance spectroscopy (EIS) to measure ion conductivity. "
        "The ion conductivity measurements will directly assess the impact on ion movement. "
        "**Defect Density Analysis:** Employ photoluminescence spectroscopy (PL) to assess the "
        "defect density, with a higher signal indicating fewer defects."
    )

    synthesis = guard_hypothesis_synthesis(report)
    analysis = guard_hypothesis_analysis(report, synthesis)
    audit = deterministic_hypothesis_audit(analysis, synthesis)
    lower = f"{synthesis}\n{analysis}".lower()

    assert "cannot isolate valence alone" in lower
    assert "factorial design" in lower
    assert "mix ionic and electronic transport" in lower
    assert "blocking/nonblocking controls" in lower
    assert "cannot alone quantify defect-state density" in lower
    assert "denser and more stable network" not in lower
    assert not audit.findings


def test_dap_guard_defines_identity_calibrates_methods_and_hides_internal_process() -> None:
    report = (
        "No direct evidence supports this proposal. DAP2+ has stronger electrostatic interactions "
        "and tighter binding than monovalent spacers. DSC is useful for assessing binding energies. "
        "ITC is applicable for studying binding processes. Impedance spectroscopy evaluates ion "
        "migration rates by analyzing electrical resistance. Background claims (B1 and B2) support "
        "the analogy.\n\n---\n\n**Note on Kantian Constraints**: The report adheres to "
        "Kantian constraints and the philosophy manager's audit process."
    )

    synthesis = guard_hypothesis_synthesis(report)
    analysis = guard_hypothesis_analysis(report, synthesis)
    audit = deterministic_hypothesis_audit(analysis, synthesis)
    lower = f"{synthesis}\n{analysis}".lower()

    assert "propane-1,3-diammonium" in lower
    assert "c3h12n2" in lower
    assert "cannot isolate valence alone" in lower
    assert "does not directly assign a spacer-lattice binding energy" in lower
    assert "cannot directly measure binding within an already formed solid" in lower
    assert "cannot alone yield an ion-migration rate" in lower
    assert "b1" not in lower and "b2" not in lower
    assert "kantian" not in lower and "philosophy manager" not in lower
    assert not audit.findings


def test_guard_removes_proxy_overclaims_and_singular_constraint_leak() -> None:
    report = (
        "No direct evidence supports this proposal. Narrower XRD peaks suggest higher crystallinity, "
        "with narrower peaks indicating higher crystallinity. PL can estimate defect density and "
        "higher values suggest fewer defects. SEM provides visual confirmation of the crystallinity "
        "and defect levels. The rationale behind these specific tests is that they directly measure "
        "the parameters predicted by the hypothesis.\n\nThis report aligns with the Kantian "
        "constraint checking used by the philosophy manager."
    )

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    audit = deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "narrower peaks suggest higher crystallinity" not in lower
    assert "higher values suggest fewer defects" not in lower
    assert "visual confirmation of the crystallinity and defect levels" not in lower
    assert "directly measure the parameters predicted" not in lower
    assert "kantian" not in lower and "philosophy manager" not in lower
    assert not audit.findings


def test_guard_calibrates_tof_sims_and_global_design_endorsement() -> None:
    report = (
        "No direct evidence supports the proposal. Ion migration rates will be assessed using "
        "time-of-flight secondary ion mass spectrometry (ToF-SIMS), which measures the movement "
        "of ions through the perovskite structure over time. The proposed tests are well-designed "
        "and appropriate for evaluating the predictions. These methods align with the properties "
        "being investigated and can provide quantitative data to support or refute the hypothesis."
    )

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    audit = deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "measures the movement of ions" not in lower
    assert "well-designed and appropriate" not in lower
    assert "methods align with the properties" not in lower
    assert "maps ion distributions" in lower
    assert not audit.findings


def test_afm_boundary_and_later_design_endorsement_do_not_false_pass() -> None:
    report = (
        "No direct evidence supports the proposal. AFM measures interlayer morphology. AFM does "
        "not directly measure electronic defect density, so an independent defect-sensitive "
        "measurement is required. The proposed tests are well-designed to provide specific, "
        "measurable data, but the identity and measurement gaps remain unresolved."
    )

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    audit = deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "well-designed to provide" not in lower
    assert "address distinct observables" in lower
    assert not audit.findings


def test_guard_removes_pce_direction_and_trpl_transport_proxy() -> None:
    report = (
        "No direct evidence supports this proposal. Incorporating DAP2+ is expected to yield a "
        "higher PCE compared to monovalent spacers, with a directional increase. Time-resolved "
        "photoluminescence (TRPL) will measure carrier lifetime. This will help substantiate the "
        "hypothesis regarding decreased trap density and enhanced charge carrier mobility. X-ray "
        "diffraction (XRD) analysis can evaluate crystallinity and grain size."
    )

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    audit = deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "higher pce" not in lower
    assert "without a prespecified direction" in lower
    assert "cannot alone assign trap density or mobility" in lower
    assert "not equivalent to film grain size" in lower
    assert not audit.findings


def test_pea_identity_novelty_structure_and_comparison_are_calibrated() -> None:
    report = (
        "No direct evidence supports this proposal. The hypothesis introduces a novel chemical "
        "species (PEA+) whose linear structure causes less steric hindrance compared to bulkier "
        "divalent spacers and therefore higher mobility. It is highly testable."
    )

    guarded = guard_hypothesis_analysis(report, NO_DIRECT_CONTEXT)
    audit = deterministic_hypothesis_audit(guarded, NO_DIRECT_CONTEXT)
    lower = guarded.lower()

    assert "phenethylammonium" in lower and "c8h12n" in lower
    assert "novel chemical species" not in lower
    assert "linear structure" not in lower
    assert "cannot isolate valence alone" in lower
    assert "highly testable" not in lower
    assert not audit.findings
