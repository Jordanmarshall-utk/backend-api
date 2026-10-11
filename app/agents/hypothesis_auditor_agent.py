"""Evidence-aware quality auditing for hypothesis evaluation reports."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field


Severity = Literal["error", "warning", "info"]
Verdict = Literal["pass", "revise", "block"]


class AuditFinding(BaseModel):
    id: str = Field(default_factory=lambda: f"finding-{uuid.uuid4().hex[:10]}")
    severity: Severity
    category: str
    claim: str = ""
    issue: str
    required_change: str


class HypothesisAudit(BaseModel):
    audit_id: str = Field(default_factory=lambda: f"audit-{uuid.uuid4().hex[:12]}")
    verdict: Verdict = "pass"
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    findings: list[AuditFinding] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    summary: str = ""
    improvement_proposals: list[str] = Field(default_factory=list)
    source: str = "deterministic"


LLMCallable = Callable[[str], str]


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_validate(model_type, value):
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)
    return model_type.parse_obj(value)


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return normalized.replace("−", "-").replace("–", "-")


def _finding(
    severity: Severity,
    category: str,
    issue: str,
    required_change: str,
    claim: str = "",
) -> AuditFinding:
    return AuditFinding(
        severity=severity,
        category=category,
        claim=claim[:500],
        issue=issue,
        required_change=required_change,
    )


def deterministic_hypothesis_audit(
    report: str,
    hypothesis_context: str = "",
) -> HypothesisAudit:
    """Check high-risk scientific claims without suppressing unsupported novelty."""
    text = _plain(report)
    context = _plain(hypothesis_context)
    lower = text.lower()
    context_lower = context.lower()
    combined = f"{context_lower}\n{lower}"
    findings: list[AuditFinding] = []

    invalid_cha = bool(
        re.search(r"cha\s*\+?\s*\(\s*pbi4\s*\)\s*2", combined, re.IGNORECASE)
    )
    charge_gate_present = any(
        marker in lower
        for marker in (
            "not charge-balanced",
            "charge balance must be resolved",
            "chemically invalid as written",
            "requires clarification",
        )
    )
    if invalid_cha and not charge_gate_present:
        findings.append(
            _finding(
                "error",
                "chemical_identity",
                "The CHA/PbI4 composition is accepted without a charge-balance gate.",
                "Treat the composition as unresolved and require an unambiguous cation identity, charge state, and charge-balanced formula before structure-based reasoning.",
                "CHA+(PbI4)2",
            )
        )

    ambiguous_3_apb = "3-apb+" in combined or "3-apb +" in combined
    identity_gate_present = all(
        token in lower for token in ("formula", "structure", "protonation")
    ) or any(
        marker in lower
        for marker in (
            "identity must be clarified",
            "unambiguous chemical identity",
            "chemically defined",
        )
    )
    if ambiguous_3_apb and not identity_gate_present:
        findings.append(
            _finding(
                "error",
                "chemical_identity",
                "3-APB+ is used as though it identifies one established chemical species.",
                "Preserve the proposed molecule as a novel candidate, but request a systematic name, formula, structure, protonation state, and charge before evaluating its structure-specific mechanism.",
                "3-APB+",
            )
        )

    cha_assignment = "cyclohexylammonium" in lower or "c6h14n" in lower
    cha_uncertain = any(
        marker in combined
        for marker in (
            "cha+ may be uncertain",
            "identity of cha+ are uncertain",
            "identity of cha+ is uncertain",
            "cha+ (identity unresolved)",
            "cha identity boundary",
            "cha must not be assigned",
        )
    )
    if cha_assignment and cha_uncertain:
        findings.append(
            _finding(
                "error",
                "chemical_identity",
                "The report assigns CHA to cyclohexylammonium after the source context marks the identity as uncertain.",
                "Retain CHA as an unresolved novel candidate and require a systematic name, formula, structure, protonation state, and net charge before structure-specific reasoning.",
                "CHA+ = cyclohexylammonium",
            )
        )

    percentages = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\s*%", text)))
    predicted_energy = sorted(
        set(
            re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:meV|eV)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    quantitative_targets = [*percentages, *predicted_energy]
    if quantitative_targets:
        findings.append(
            _finding(
                "warning",
                "unsupported_quantification",
                "The report retains precise numerical effect targets without a demonstrated evidence or power-analysis basis.",
                "Remove the numerical target or label it explicitly as a preregistered exploratory threshold with a stated derivation.",
                ", ".join(quantitative_targets),
            )
        )

    method_rules = (
        (
            ("xrd", "d-spacing"),
            ("direct evidence", "strong evidence", "demonstrate", "prove"),
            "measurement_validity",
            "A change in XRD spacing alone is treated as direct proof of stronger interlayer bonding.",
            "Describe XRD as structural evidence and require an independent energetic or interaction-sensitive measurement for bonding-strength claims.",
        ),
        (
            ("afm", "defect density"),
            ("assess", "measure", "direct"),
            "measurement_validity",
            "AFM morphology is treated as a direct measurement of electronic defect density.",
            "Use AFM for morphology and add a defect-sensitive measurement before making defect-density claims.",
        ),
        (
            ("uv-vis", "interface stability"),
            ("assess", "measure", "confirm", "demonstrate"),
            "measurement_validity",
            "UV-Vis is treated as a direct test of interface stability.",
            "Use UV-Vis for optical absorption or band-edge changes and add structural, chemical, or operational measurements for interface stability.",
        ),
        (
            ("steady-state pl", "lifetime"),
            ("measure", "determine", "confirm"),
            "measurement_validity",
            "Steady-state PL is treated as a direct carrier-lifetime measurement.",
            "Use time-resolved PL or another time-resolved method for carrier-lifetime claims.",
        ),
        (
            ("tof-sims", "migration rate"),
            ("direct evidence", "measure", "measuring", "quantify", "suitable"),
            "measurement_validity",
            "ToF-SIMS depth profiles are treated as direct measurements of an ion-migration rate.",
            "Use a controlled time, bias, or temperature series and complementary transient or impedance measurements before assigning a migration rate.",
        ),
    )
    for required_terms, assertion_terms, category, issue, change in method_rules:
        if required_terms == ("afm", "defect density"):
            afm_boundary = bool(
                "afm does not directly measure electronic defect density" in lower
                or "afm for morphology" in lower
            )
            if afm_boundary:
                continue
        if required_terms == ("tof-sims", "migration rate"):
            tof_boundary = bool(
                "tof-sims maps stress-dependent ion distributions" in lower
                and "migration rate requires" in lower
            )
            tof_direct_body_claim = bool(
                re.search(
                    r"ToF-SIMS[^.]{0,160}(?:measures?\s+the\s+movement|"
                    r"directly\s+measures?|quantif(?:y|ies)\s+(?:an?\s+)?ion-migration\s+rate)",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            if tof_boundary and not tof_direct_body_claim:
                continue
        if all(term in combined for term in required_terms) and any(
            term in combined for term in assertion_terms
        ):
            findings.append(
                _finding("warning", category, issue, change, " / ".join(required_terms))
            )

    dap_mentioned = bool("dap2+" in combined or re.search(r"\bdap\b", combined))
    dap_identity_defined = bool(
        ("propane-1,3-diammonium" in lower or "1,3-propanediammonium" in lower)
        and "c3h12n2" in lower
        and "ambiguous acronym" in lower
    )
    if dap_mentioned and not dap_identity_defined:
        findings.append(
            _finding(
                "error",
                "chemical_identity",
                "DAP2+ is used for structure-specific reasoning without a complete, unambiguous identity boundary.",
                "Provisionally define the intended candidate as propane-1,3-diammonium (C3H12N2^2+) only if that is intended, and require confirmation of its systematic name, structure, counterion, protonation state, and phase stoichiometry.",
                "DAP2+",
            )
        )

    pea_mentioned = "pea+" in lower or "phenethylammonium" in lower
    pea_identity_defined = bool("phenethylammonium" in lower and "c8h12n" in lower)
    if pea_mentioned and not pea_identity_defined:
        findings.append(
            _finding(
                "error",
                "chemical_identity",
                "PEA+ is used for structure-specific reasoning without a complete identity boundary.",
                "Define PEA+ as phenethylammonium (C8H12N+), specify its protonation state, counterion, and phase stoichiometry, and separate that identity from unsupported property claims.",
                "PEA+",
            )
        )

    divalent_valence_attribution = bool(
        ("eda2+" in combined or "ethane-1,2-diammonium" in combined or dap_mentioned or "double positive charge" in combined or (pea_mentioned and "divalent spacer" in combined))
        and ("monovalent" in combined or "divalent nature" in combined)
        and any(term in combined for term in ("barrier", "ion migration", "ion movement", "mobility", "charge transport"))
    )
    valence_confounding_disclosed = bool(
        "cannot isolate valence alone" in lower
        and ("factorial design" in lower or "controlled spacer series" in lower)
    )
    if divalent_valence_attribution and not valence_confounding_disclosed:
        findings.append(
            _finding(
                "warning",
                "design_confounding",
                "The dication-versus-monovalent comparison attributes an outcome to valence while changing multiple molecular variables.",
                "State that charge, ammonium-site count, molecular structure, size, flexibility, and interfacial chemistry are jointly changed; require a controlled spacer series or factorial design before making a valence-specific conclusion.",
                "dication versus monovalent spacer",
            )
        )

    pea_false_novelty = bool(
        pea_mentioned
        and re.search(r"novel\s+(?:chemical\s+)?species[^.]{0,80}PEA", text, re.IGNORECASE)
    )
    if pea_false_novelty:
        findings.append(
            _finding(
                "warning",
                "novelty_calibration",
                "Phenethylammonium is presented as a novel chemical species rather than an established spacer used in a new hypothesis.",
                "Assign novelty only to the specific comparison, mechanism, or application relative to the searched scope, not to PEA itself.",
                "novel PEA species",
            )
        )

    pea_linear_claim = bool(
        pea_mentioned and re.search(r"\blinear\s+(?:alkyl\s+chain|structure)\b", lower)
    )
    if pea_linear_claim:
        findings.append(
            _finding(
                "warning",
                "chemical_structure",
                "PEA is described as a linear alkyl spacer, omitting its phenyl group and aromatic structure.",
                "Describe PEA as phenethylammonium and treat packing or steric effects as measurements, not consequences of a generic linear-chain model.",
                "PEA linear structure",
            )
        )

    eis_direct_claim = bool(
        re.search(
            r"(?:\bEIS\b|(?:electrochemical\s+)?impedance spectroscopy)[^.]{0,180}"
            r"(?:measure|measuring|directly assess|quantif)[^.]{0,100}ion(?:ic)?\s+"
            r"(?:conductivity|migration|movement)",
            combined,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"ion\s+conductivity\s+measurements?[^.]{0,100}directly\s+assess[^.]{0,80}ion\s+movement",
            combined,
            flags=re.IGNORECASE,
        )
    )
    eis_boundary_present = all(
        marker in lower
        for marker in ("mix ionic and electronic transport", "blocking/nonblocking controls")
    )
    if eis_direct_claim:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "EIS is treated as a direct, uniquely ionic measurement of ion movement.",
                "Describe the mixed ionic/electronic, dielectric, and contact contributions; require validated equivalent-circuit analysis, blocking controls, temperature/bias series, and a complementary time-resolved ion measurement.",
                "EIS / ionic conductivity",
            )
        )

    pl_defect_claim = bool(
        re.search(r"\b(?:PL|photoluminescence)\b", combined, re.IGNORECASE)
        and re.search(r"defect(?:-state)?\s+density|fewer\s+defects", combined, re.IGNORECASE)
    )
    pl_boundary_present = bool(
        "recombination observables" in lower
        and "cannot alone quantify defect-state density" in lower
        and "independent trap-sensitive measurement" in lower
    )
    pl_direct_body_claim = bool(
        re.search(
            r"(?:estimate|assess|evaluate|measure|quantify)\s+(?:the\s+)?defect\s+density|"
            r"defect(?:-state)?\s+density[^.]{0,100}as\s+measured\s+by\s+(?:PL|photoluminescence)|"
            r"defect(?:-state)?\s+density[^.]{0,120}indicated\s+by[^.]{0,80}(?:PL|photoluminescence)|"
            r"higher\s+values\s+suggest\s+fewer\s+defects|"
            r"reduction\s+in\s+defects\s+would\s+correspond",
            lower,
        )
    )
    if pl_defect_claim and (not pl_boundary_present or pl_direct_body_claim):
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "PL intensity or lifetime is used as though it uniquely quantifies defect-state density.",
                "Use PL for recombination observables, control optical and compositional confounders, and pair it with an independent trap-sensitive measurement before assigning defect density.",
                "PL / defect-state density",
            )
        )

    no_direct_support = any(
        marker in combined
        for marker in ("no direct evidence", "lacks direct supporting evidence", "no directly relevant evidence")
    )
    directional_network_claim = bool(
        re.search(r"\bdenser(?:\s+and\s+more\s+stable)?\s+network\b", combined)
    )
    network_boundary_present = "different local interaction topology" in lower
    if no_direct_support and directional_network_claim and not network_boundary_present:
        findings.append(
            _finding(
                "warning",
                "mechanism_calibration",
                "A denser or more stable interaction network is assigned without direct structural or interaction-sensitive evidence.",
                "Replace the directional network claim with an unverified difference in local interaction topology and state the measurements needed to determine its direction.",
                "denser and more stable network",
            )
        )

    directional_binding_claim = bool(
        re.search(r"\b(?:stronger|tighter|enhanced)\b[^.]{0,100}\b(?:binding|electrostatic interactions?)\b", combined)
    )
    binding_boundary_present = bool(
        "different interfacial binding geometry" in lower
        or "proposed difference in electrostatic interactions" in lower
        or "testable difference in local interaction geometry" in lower
        or "different electrostatic interaction geometry and energetics" in lower
    )
    if no_direct_support and directional_binding_claim and not binding_boundary_present:
        findings.append(
            _finding(
                "warning",
                "mechanism_calibration",
                "A stronger or tighter binding interaction is assigned without direct interaction-sensitive evidence.",
                "Use a nondirectional, testable difference in interfacial binding geometry and identify independent structural, interaction-sensitive, and computational tests.",
                "stronger/tighter binding",
            )
        )

    lattice_parameter_increase = bool(
        re.search(r"increase\s+in\s+(?:the\s+)?lattice\s+parameter", combined)
        or re.search(r"lattice\s+parameter[^.]{0,80}\bincreas", combined)
    )
    unit_cell_volume_decrease = bool(
        re.search(r"unit[- ]cell\s+volume[^.]{0,180}\bdecreas", combined)
    )
    if lattice_parameter_increase and unit_cell_volume_decrease:
        findings.append(
            _finding(
                "warning",
                "internal_consistency",
                "The report predicts both an increased lattice parameter and a decreased unit-cell volume without defining anisotropic axes or a structural model that reconciles them.",
                "State a nondirectional change in refined cell metrics, or define the affected axes and provide a structural model that makes the directional predictions mutually consistent.",
                "lattice-parameter direction",
            )
        )

    formation_energy_binding_claim = bool(
        re.search(r"formation\s+energ[^.]{0,180}(?:stronger|tighter)[^.]{0,80}binding", combined)
        or re.search(r"(?:stronger|tighter)[^.]{0,80}binding[^.]{0,180}formation\s+energ", combined)
    )
    formation_energy_boundary_present = bool(
        "balanced, shared set of reference states" in lower
        and "would not alone establish local interaction strength" in lower
    )
    if formation_energy_binding_claim and not formation_energy_boundary_present:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "Formation energy is used as though it directly establishes local binding strength across potentially different compositions or layer topologies.",
                "Define balanced reference states and compare like-for-like structures; use an independent interaction-sensitive analysis for local binding strength.",
                "formation energy / binding strength",
            )
        )

    calorimetry_binding_claim = bool(
        (
            "dsc" in combined
            and "binding" in combined
            and "does not directly assign a spacer-lattice binding energy" not in lower
        )
        or (
            "itc" in combined
            and "binding" in combined
            and "cannot directly measure binding within an already formed solid perovskite lattice" not in lower
        )
    )
    if calorimetry_binding_claim:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "DSC or ITC is presented as a direct binding-energy measurement for a formed solid perovskite lattice.",
                "Limit DSC to thermal transitions/enthalpic events, limit ITC to a defined solution titration model, and require a thermodynamic cycle or validated atomistic calculation for a lattice binding-energy claim.",
                "DSC / ITC / binding energy",
            )
        )

    internal_process_leak = bool(
        re.search(
            r"\b(?:Kantian|Humean) constraint(?:s|\s+checking)?\b|"
            r"\bphilosophy manager\b|\breport adheres to\b",
            lower,
        )
    )
    if internal_process_leak:
        findings.append(
            _finding(
                "error",
                "internal_process_leak",
                "The user-facing scientific report exposes internal reasoning-agent or audit-process language.",
                "Remove all references to philosophy agents, named reasoning constraints, prompts, and internal audit compliance from the report body.",
                "internal reasoning-process note",
            )
        )

    bare_background_ids = sorted(set(re.findall(r"\b(?:B|KG)\d+\b", text, flags=re.IGNORECASE)))
    if no_direct_support and bare_background_ids:
        findings.append(
            _finding(
                "warning",
                "provenance",
                "The zero-direct-evidence report retains background or knowledge-graph anchor identifiers.",
                "Remove background/KG identifiers from the report; describe any analogy generically and do not present it as validation.",
                ", ".join(bare_background_ids),
            )
        )

    xrd_fwhm_overclaim = bool(
        re.search(r"\b(?:narrower|sharper)\s+peaks?[^.]{0,100}\b(?:higher|increase\w*)\s+crystallinity", lower)
        or re.search(r"\bFWHM\b[^.]{0,140}\bindicat\w*\s+higher\s+crystallinity", text, re.IGNORECASE)
    )
    xrd_boundary_present = "fwhm alone does not establish a directional change in crystallinity" in lower
    if xrd_fwhm_overclaim or ("fwhm" in lower and "crystallinity" in lower and not xrd_boundary_present):
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "XRD peak width is treated as a standalone directional measure of crystallinity.",
                "Require deconvolution of coherent-domain size, microstrain, texture, phase overlap, and instrument broadening; do not infer crystallinity from FWHM alone.",
                "XRD FWHM / crystallinity",
            )
        )

    sem_direct_claim = bool(
        re.search(r"\bSEM\b[^.]{0,180}\b(?:confirm|measure|visual confirmation)[^.]{0,100}\b(?:crystallinity|defect levels?)", text, re.IGNORECASE)
    )
    sem_boundary_present = "without directly measuring crystallinity or electronic defect levels" in lower
    if sem_direct_claim and not sem_boundary_present:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "SEM morphology is treated as direct confirmation of crystallinity or electronic defect levels.",
                "Limit SEM to surface/cross-sectional morphology and apparent grain size; use diffraction and defect-sensitive measurements for the other claims.",
                "SEM / crystallinity / defect levels",
            )
        )

    lowered_activation_barrier = bool(
        re.search(
            r"\b(?:reduce|reduces|reduced|reducing|lower|lowers|lowered|lowering)\s+"
            r"(?:the\s+)?(?:(?:activation\s+)?energy(?:\s+required)?|activation\s+barrier)\s+"
            r"for\s+ion\s+migration\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    migration_suppression = bool(
        re.search(
            r"\b(?:suppress(?:es|ed|ing)?\s+ion\s+migration|"
            r"suppression\s+of\s+ion\s+migration|"
            r"reduc(?:e|es|ed|ing)\s+ion\s+migration|"
            r"reduction\s+of\s+ion\s+migration|"
            r"stability\s+against\s+ion\s+migration)\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    if lowered_activation_barrier and migration_suppression:
        findings.append(
            _finding(
                "error",
                "mechanistic_consistency",
                "The hypothesis claims migration suppression while lowering the activation energy or barrier for migration.",
                "Reverse the mechanistic direction: suppression of a thermally activated pathway requires an increased effective activation barrier, then identify measurements that can test that barrier.",
                "lower migration activation energy",
            )
        )

    trpl = r"(?:time-resolved\s+photoluminescence(?:\s+decay)?(?:\s+experiments?)?|TRPL)"
    trpl_rate_matches = [
        *re.finditer(
            rf"(?:{trpl})[^.]{{0,180}}(?:quantif\w*|measur\w*|reflect\w*)[^.]{{0,80}}"
            r"(?:ion\s+migration|migration\s+rates?)",
            combined,
            flags=re.IGNORECASE,
        ),
        *re.finditer(
            rf"(?:ion\s+migration|migration\s+rates?)[^.]{{0,180}}(?:measured\s+by|using)\s+{trpl}",
            combined,
            flags=re.IGNORECASE,
        ),
    ]

    def negates_trpl_rate_claim(match: re.Match) -> bool:
        return bool(
            re.search(
                r"\b(?:not\s+(?:an?\s+)?(?:direct\s+)?"
                r"(?:ion[-\s]+migration(?:[-\s]+rates?)?|migration[-\s]+rates?)|"
                r"(?:cannot|can't|does\s+not|do\s+not|is\s+not|are\s+not)"
                r"[^.]{0,80}(?:quantif\w*|measur\w*|reflect\w*))",
                match.group(0),
                flags=re.IGNORECASE,
            )
        )

    trpl_rate_claim = any(
        not negates_trpl_rate_claim(match) for match in trpl_rate_matches
    )
    if trpl_rate_claim:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "Time-resolved photoluminescence is treated as a direct ion-migration-rate measurement.",
                "Use TRPL for carrier recombination dynamics and add dedicated bias-, time-, and temperature-dependent ion-transport measurements for migration kinetics.",
                "TRPL / ion migration rate",
            )
        )

    grain_boundary_claim = bool(
        "grain boundary density" in combined
        and "xrd" in combined
        and "sem" in combined
        and "do not directly quantify grain-boundary density" not in combined
    )
    if grain_boundary_claim:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "XRD and surface SEM are presented as direct measures of grain-boundary density.",
                "Limit XRD to phase/average crystallinity and SEM to surface morphology/apparent grain size; use dedicated cross-sectional or three-dimensional microscopy for grain-boundary density.",
                "XRD / SEM / grain-boundary density",
            )
        )

    pce_overclaim = (
        ("pce" in combined or "power conversion efficiency" in combined)
        and (
            "ion migration" in combined
            or any(
                marker in combined
                for marker in (
                    "no direct evidence",
                    "lacks direct supporting evidence",
                    "no directly relevant evidence",
                )
            )
        )
        and any(
            marker in combined
            for marker in (
                "projected increase",
                "expected increase",
                "expected to translate into improved",
                "expected to yield a higher pce",
                "predicted to yield a higher pce",
                "will increase measurably",
                "should be higher",
                "increase in pce would support",
                "increase in the power conversion efficiency",
                "improvement in pce",
                "improvements in efficiency",
                "expected to be measurably higher",
                "expected to be higher",
            )
        )
        and "secondary outcome without a prespecified direction" not in combined
    )
    if pce_overclaim:
        findings.append(
            _finding(
                "warning",
                "evidence_calibration",
                "Improved PCE is predicted as a consequence of the proposed ion-migration mechanism without direct support.",
                "Treat PCE as a secondary device-level outcome without a prespecified direction and validate ion transport independently.",
                "projected PCE improvement",
            )
        )

    generic_sims_rate_claim = bool(
        re.search(r"\b(?:SIMS|secondary\s+ion\s+mass\s+spectrometry)\b", text, re.IGNORECASE)
        and re.search(r"direct\s+evidence\s+of\s+ion\s+migration\s+rates?|directly\s+measures?\s+"
                      r"(?:an?\s+)?ion[- ]migration\s+rate", text, re.IGNORECASE)
    )
    generic_sims_boundary = bool(
        "sims maps composition or stress-induced ion redistribution" in lower
        and "neither alone directly measures an ion-migration rate" in lower
    )
    if generic_sims_rate_claim and not generic_sims_boundary:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "SIMS or TR-XRD is presented as a direct measurement of an ion-migration rate.",
                "Use controlled time/bias/temperature series, calibrated depth profiles, and a complementary transient or impedance measurement before assigning kinetics.",
                "SIMS / TR-XRD / ion-migration rate",
            )
        )

    xrd_sem_proxy_claim = bool(
        re.search(r"\bXRD\b|X-ray\s+diffraction", text, re.IGNORECASE)
        and re.search(r"\bSEM\b|scanning\s+electron\s+microscopy", text, re.IGNORECASE)
        and re.search(r"improved\s+crystallinity|fewer\s+defects|defect\s+levels?", text, re.IGNORECASE)
    )
    xrd_sem_boundary = bool(
        "xrd and sem can assess phase, texture, peak shape, and morphology" in lower
        and "do not alone establish improved crystallinity" in lower
    )
    if xrd_sem_proxy_claim and not xrd_sem_boundary:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "XRD and SEM are used as direct evidence of improved crystallinity or fewer electronic defects.",
                "Limit XRD/SEM to phase, texture, peak-shape, and morphology observables and add an independent trap-sensitive measurement for electronic defect claims.",
                "XRD / SEM / crystallinity and defects",
            )
        )

    no_interlayer_control = bool(
        re.search(r"controls?\s+(?:samples?\s+)?without\s+(?:PEA|the\s+(?:PEA\s+)?interlayer)", text, re.IGNORECASE)
    )
    no_interlayer_boundary = "cannot isolate the pea mechanism" in lower
    if no_interlayer_control and not no_interlayer_boundary:
        findings.append(
            _finding(
                "warning",
                "design_confounding",
                "A PEA-interlayer versus no-interlayer comparison changes composition and processing and cannot isolate the proposed molecular mechanism.",
                "Add sham-processed, thickness-matched, and alternative-spacer controls and verify the resulting interfacial phase.",
                "PEA versus no-interlayer control",
            )
        )

    tpv_tpc_proxy_claim = bool(
        ("tpv" in combined or "transient photovoltage" in combined)
        and any(term in combined for term in ("mobility", "mobilities", "carrier lifetime", "charge transport efficiency"))
    )
    tpv_tpc_boundary = "do not independently measure intrinsic carrier lifetime or mobility" in lower
    if tpv_tpc_proxy_claim and not tpv_tpc_boundary:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "TPV and TPC transients are treated as direct measurements of intrinsic carrier lifetime or mobility.",
                "Interpret TPV/TPC as device-level responses coupled to recombination, extraction, capacitance, contacts, and operating point; add an independent transport method for mobility.",
                "TPV / TPC / lifetime or mobility",
            )
        )

    malformed_fapbi3 = bool(re.search(r"\bFAPB3\b", text))
    if malformed_fapbi3:
        findings.append(
            _finding(
                "error",
                "chemical_identity",
                "FAPB3 is not an unambiguous formula for formamidinium lead iodide and is inconsistent with FAPbI3 elsewhere in the report.",
                "Use FAPbI3 consistently and preserve element capitalization and iodine stoichiometry.",
                "FAPB3",
            )
        )

    novelty_from_absence = bool(
        no_direct_support
        and re.search(r"appears\s+novel[^.]{0,160}(?:no|lack|without)\s+direct", combined)
    )
    if novelty_from_absence:
        findings.append(
            _finding(
                "warning",
                "novelty_calibration",
                "Novelty is inferred from the absence of direct results in the retrieved packet.",
                "State that novelty is undetermined and requires a broader prior-art search; an empty retrieval is not evidence of novelty.",
                "novelty from retrieval absence",
            )
        )

    trpl_proxy_overclaim = bool(
        re.search(r"\bTRPL\b|time-resolved\s+photoluminescence", text, re.IGNORECASE)
        and re.search(r"trap\s+density|charge\s+carrier\s+mobility", text, re.IGNORECASE)
        and "does not alone quantify trap-state density or charge-carrier mobility" not in lower
    )
    if trpl_proxy_overclaim:
        findings.append(
            _finding(
                "warning",
                "measurement_validity",
                "TRPL recombination dynamics are treated as direct measurements of trap density or mobility.",
                "Limit TRPL to recombination dynamics and add independent trap-sensitive and transport measurements.",
                "TRPL / trap density / mobility",
            )
        )

    design_overclaim = any(
        term in lower
        for term in (
            "tests are well-designed",
            "design is robust",
            "methods are fully appropriate",
            "tests are well-aligned",
            "tests outlined are robust and well-suited",
            "tests are robust and well-suited",
            "proposed tests are well-suited",
        )
    )
    unresolved_gaps = any(
        term in lower
        for term in (
            "only partially aligned",
            "definition-required",
            "identity must remain unresolved",
            "not ready for mechanistic evaluation",
            "measurement gaps",
            "no directly relevant evidence",
            "lacks direct supporting evidence",
        )
    )
    if design_overclaim and unresolved_gaps:
        findings.append(
            _finding(
                "warning",
                "internal_consistency",
                "The report praises the test design after documenting unresolved identity or measurement gaps.",
                "Replace the global endorsement with a method-by-method statement of what each test can establish and what additional controls remain necessary.",
            )
        )

    no_direct_evidence = any(
        marker in lower
        for marker in (
            "no direct evidence",
            "does not directly",
            "absence of direct",
            "indirect evidence",
        )
    )
    overclaim_terms = (
        "the literature validates",
        "the evidence validates",
        "is confirmed by the literature",
        "fully substantiates",
        "directly establishes",
    )
    if no_direct_evidence and any(term in lower for term in overclaim_terms):
        findings.append(
            _finding(
                "error",
                "evidence_calibration",
                "The report simultaneously acknowledges missing direct evidence and presents the hypothesis as validated.",
                "Separate retrieved evidence, analogy, inference, and prediction; retain the hypothesis as speculative but testable.",
            )
        )

    report_evidence_ids = set(re.findall(r"\bE\d+\b", text))
    context_evidence_ids = set(re.findall(r"\bE\d+\b", context))
    ungrounded_ids = sorted(report_evidence_ids - context_evidence_ids)
    if context.strip() and ungrounded_ids:
        findings.append(
            _finding(
                "error",
                "provenance",
                "The report cites evidence identifiers that are absent from the supplied context.",
                "Remove the identifiers or link them to evidence records that were actually supplied to the analysis.",
                ", ".join(ungrounded_ids),
            )
        )

    if "novel" in lower and no_direct_evidence:
        strengths = [
            "The report permits a novel proposal while acknowledging that direct validation is absent."
        ]
    else:
        strengths = []
    if any(term in lower for term in ("testable", "falsifiable", "control")):
        strengths.append("The report frames at least part of the proposal as experimentally testable.")

    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    verdict: Verdict = "block" if errors else "revise" if warnings else "pass"
    score = max(0.0, round(1.0 - (0.28 * errors) - (0.12 * warnings), 2))
    summary = (
        "No deterministic scientific-report issues were detected."
        if not findings
        else f"Detected {errors} blocking and {warnings} revision-level issue(s)."
    )
    return HypothesisAudit(
        verdict=verdict,
        score=score,
        findings=findings,
        strengths=strengths,
        summary=summary,
        source="deterministic",
    )


def _extract_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Auditor response did not contain a JSON object")


def _audit_prompt(report: str, hypothesis_context: str) -> str:
    return f"""You are the live scientific auditor for a hypothesis evaluation report.

Audit the report without suppressing novelty. A hypothesis may introduce a new molecule or mechanism, but unknown identity, missing direct evidence, and predictions must be labeled correctly. Distinguish retrieved evidence from analogy, inference, and untested prediction. Check chemistry and charge balance, evidence provenance, unsupported numerical targets, whether each measurement can support its stated claim, and whether the conclusion overstates validation.

Return one JSON object only with this schema:
{{
  "verdict": "pass|revise|block",
  "score": 0.0,
  "findings": [
    {{
      "severity": "error|warning|info",
      "category": "short_category",
      "claim": "quoted or summarized claim",
      "issue": "specific scientific issue",
      "required_change": "bounded report correction"
    }}
  ],
  "strengths": ["specific strength"],
  "summary": "concise audit summary",
  "improvement_proposals": ["optional source-code or prompt improvement proposal for human approval"]
}}

Do not propose autonomous source-code edits. Do not invent citations or facts.

HYPOTHESIS AND EVIDENCE CONTEXT
{hypothesis_context[-18000:]}

REPORT
{report[-18000:]}
"""


def _merge_findings(
    deterministic: list[AuditFinding], llm_findings: list[AuditFinding]
) -> list[AuditFinding]:
    merged: list[AuditFinding] = []
    seen: set[tuple[str, str]] = set()
    for item in [*deterministic, *llm_findings]:
        key = (item.category.strip().lower(), item.issue.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


class HypothesisAuditorAgent:
    """Combines deterministic safety checks with an optional evidence-focused LLM audit."""

    def __init__(self, llm_callable: Optional[LLMCallable] = None):
        self.llm_callable = llm_callable

    def audit(self, report: str, hypothesis_context: str = "") -> HypothesisAudit:
        base = deterministic_hypothesis_audit(report, hypothesis_context)
        if self.llm_callable is None:
            return base

        try:
            raw = self.llm_callable(_audit_prompt(report, hypothesis_context))
            payload = _extract_json_object(str(raw))
            payload.setdefault("audit_id", base.audit_id)
            payload.setdefault("source", "humean_evidence_llm")
            llm_audit = _model_validate(HypothesisAudit, payload)
        except Exception:
            return base

        findings = _merge_findings(base.findings, llm_audit.findings)
        errors = sum(item.severity == "error" for item in findings)
        warnings = sum(item.severity == "warning" for item in findings)
        verdict: Verdict = "block" if errors else "revise" if warnings else llm_audit.verdict
        if verdict == "pass" and findings:
            verdict = "revise"
        score = min(base.score, llm_audit.score) if findings else llm_audit.score
        return HypothesisAudit(
            audit_id=base.audit_id,
            verdict=verdict,
            score=score,
            findings=findings,
            strengths=list(dict.fromkeys([*base.strengths, *llm_audit.strengths])),
            summary=llm_audit.summary or base.summary,
            improvement_proposals=list(dict.fromkeys(llm_audit.improvement_proposals)),
            source="deterministic+humean_evidence_llm",
        )


__all__ = [
    "AuditFinding",
    "HypothesisAudit",
    "HypothesisAuditorAgent",
    "deterministic_hypothesis_audit",
]
