"""Deterministic quality controls for hypothesis retrieval and reporting."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


_RETRY_SECTION_RE = re.compile(
    r"(?im)^\s*(?:retry requirement|retry request|retry guidance|generation guidance|model instructions?)\s*:"
)
_RETRY_SENTENCE_RE = re.compile(
    r"(?i)\bNovel compounds,\s*variables,\s*and mechanisms are allowed(?:\s+and encouraged)?\b"
)
_CONTINUATION_PREFIX_RE = re.compile(
    r"(?is)^\s*How can we continue exploring this hypothesis\?\s*"
)
_RETRIEVAL_DISCLOSURE_PREFIXES = (
    re.compile(
        r"(?is)^\s*Exploratory hypothesis\s*"
        r"\(\s*not directly supported by (?:the\s+)?current local packet\s*\)\s*:\s*"
    ),
    re.compile(
        r"(?is)^\s*Unverified continuation\s*"
        r"\(\s*no direct supporting evidence retrieved\s*\)\s*:\s*"
    ),
)
_RETRIEVAL_DISCLOSURE_SUFFIX_RE = re.compile(
    r"(?is)\s*(?:#{2,}\s*)?Any implied prior support was omitted because "
    r"no DOI or paper_slug was supplied\.?\s*$"
)
_RETRIEVAL_GENERATED_SECTION_RE = re.compile(
    r"(?is)\s+#{2,}\s+(?:Evidence-grounded Revision|Socratic Pass)\b.*$"
)
_RETRIEVAL_EVIDENCE_ANCHOR_RE = re.compile(
    r"(?is)\s+(?:#{1,6}\s*)?Evidence anchor\s*:.*$"
)
_METHOD_ACRONYMS = {
    "AFM", "DJ", "DOI", "DSC", "DTA", "EDX", "EIS", "EQE", "FF", "FTIR", "GIWAXS",
    "I-V", "IMP", "ITC", "J-V", "JSC", "MPP", "NMR", "PCE", "PL", "PLQY", "RH",
    "RP", "SEM", "TAS", "TEM", "TGA", "TOF", "TOF-SIMS", "TPV", "TRMC", "TRPL", "UV",
    "VOC", "XPS", "XRD",
}


def _collapse(value: str) -> str:
    return " ".join(str(value or "").split())


def _plain_scientific_text(value: str) -> str:
    value = re.sub(r"(?i)\[(?:E|B)\d+\]", " ", str(value or ""))
    value = re.sub(r"(?i)\(\s*DOI\s*:[^)]+\)", " ", value)
    value = re.sub(r"(?i)\bDOI\s*:\s*10\.\S+", " ", value)
    value = re.sub(r"[`*_#>]", " ", value)
    value = _collapse(value)
    if len(value) <= 3000:
        return value
    shortened = value[:3000]
    boundary = max(shortened.rfind(". "), shortened.rfind("? "), shortened.rfind("; "))
    return shortened[: boundary + 1].strip() if boundary >= 600 else shortened.strip()


def _strip_retrieval_disclosures(value: str) -> tuple[str, list[str]]:
    """Remove POLARIS provenance labels without deleting the scientific claim."""
    cleaned = str(value or "").strip()
    transformations: list[str] = []
    for pattern in _RETRIEVAL_DISCLOSURE_PREFIXES:
        updated, count = pattern.subn("", cleaned, count=1)
        if count:
            cleaned = updated.strip()
            transformations.append("removed_evidence_disclosure_prefix")
            break
    updated, count = _RETRIEVAL_DISCLOSURE_SUFFIX_RE.subn("", cleaned, count=1)
    if count:
        cleaned = updated.strip()
        transformations.append("removed_evidence_disclosure_suffix")
    updated, count = _RETRIEVAL_GENERATED_SECTION_RE.subn("", cleaned, count=1)
    if count:
        cleaned = updated.strip()
        transformations.append("removed_generated_revision_section")
    updated, count = _RETRIEVAL_EVIDENCE_ANCHOR_RE.subn("", cleaned, count=1)
    if count:
        cleaned = updated.strip()
        transformations.append("removed_evidence_anchor_section")
    return cleaned, transformations


def prepare_retrieval_query(query: str) -> dict[str, Any]:
    """Separate scientific content from UI/model instructions and branch history."""
    original = str(query or "").replace("\r\n", "\n").strip()
    scientific = original
    transformations: list[str] = []

    retry_match = _RETRY_SECTION_RE.search(scientific)
    if retry_match:
        scientific = scientific[: retry_match.start()].rstrip()
        transformations.append("removed_retry_instruction_section")
    else:
        retry_sentence = _RETRY_SENTENCE_RE.search(scientific)
        if retry_sentence:
            scientific = scientific[: retry_sentence.start()].rstrip()
            transformations.append("removed_inline_generation_instruction")

    candidate = scientific
    selected = re.search(r"(?is)\bSelected continuation\s*:\s*(.+)$", scientific)
    if selected:
        candidate = re.split(r"(?m)^\s*#{2,}\s*$", selected.group(1), maxsplit=1)[0]
        transformations.append("selected_continuation_only")
    else:
        revised = re.search(
            r"(?is)\bRevised Hypothesis\s*:\s*(?:\*\*Hypothesis\s*:\*\*)?\s*(.+?)"
            r"(?=\n\s*\*\*(?:Evidence boundary|Novel proposal|Predictions|Tests)\b|\Z)",
            scientific,
        )
        if revised:
            candidate = revised.group(1)
            transformations.append("revised_hypothesis_only")
        elif _CONTINUATION_PREFIX_RE.match(scientific):
            tail = _CONTINUATION_PREFIX_RE.sub("", scientific, count=1).strip()
            hypothesis_block = re.match(
                r"(?is)(?:\*\*)?Hypothesis\s*:(?:\*\*)?\s*(.+?)"
                r"(?=\n\s*(?:\*\*)?(?:Predictions|Tests|Evidence boundary)\s*:|\Z)",
                tail,
            )
            if hypothesis_block:
                candidate = hypothesis_block.group(1)
                transformations.append("hypothesis_statement_only")
            else:
                first_question = re.match(
                    r"(?is)(.+?\?)(?=\s*\n\s*\n|\s*\*\*Evidence anchor\s*:|\Z)", tail
                )
                if first_question:
                    candidate = first_question.group(1)
                    transformations.append("continuation_question_only")
                else:
                    candidate = tail
                    transformations.append("removed_continuation_wrapper")

    candidate, disclosure_transformations = _strip_retrieval_disclosures(candidate)
    transformations.extend(disclosure_transformations)
    effective = _plain_scientific_text(candidate)
    if effective.count("?") >= 3:
        first_question_end = effective.find("?") + 1
        if first_question_end >= 30:
            effective = effective[:first_question_end].strip()
            transformations.append("first_generated_question_only")
    if len(effective) < 12:
        effective = _plain_scientific_text(scientific or original)
        transformations.append("short_query_fallback")

    return {
        "original_query": original,
        "retrieval_query": effective,
        "query_transformations": transformations,
        "query_hash": hashlib.sha256(original.encode("utf-8", errors="ignore")).hexdigest()[:16],
    }


def _claim_texts(claims: Iterable[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("claim") or "") for item in claims)


def _near(text: str, left: str, right: str, span: int = 180) -> bool:
    return bool(
        re.search(rf"(?:{left}).{{0,{span}}}(?:{right})", text, re.I | re.S)
        or re.search(rf"(?:{right}).{{0,{span}}}(?:{left})", text, re.I | re.S)
    )


def build_quality_flags(
    source_text: str,
    direct_claims: Iterable[dict[str, Any]] = (),
    background_claims: Iterable[dict[str, Any]] = (),
    strong_evidence_count: int | None = None,
) -> list[dict[str, str]]:
    """Return explainable flags; novelty is permitted and never rejected by itself."""
    text = str(source_text or "")
    direct = list(direct_claims)
    background = list(background_claims)
    direct_text = _claim_texts(direct)
    all_evidence_text = direct_text + "\n" + _claim_texts(background)
    flags: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(code: str, severity: str, message: str) -> None:
        if code not in seen:
            seen.add(code)
            flags.append({"code": code, "severity": severity, "message": message})

    if strong_evidence_count == 0:
        add(
            "no_direct_support",
            "boundary",
            "No directly relevant evidence was retrieved. Continue freely as a novel proposal, but do not use background items as validation.",
        )

    if re.search(r"(?i)methylammonium\s+stearate\s*\(\s*MSG\s*\)", text):
        add(
            "msg_identity_conflict",
            "error",
            "MSG is misidentified here. In the cited MCH/MACl system it denotes methyl-substituted guanidine, not methylammonium stearate.",
        )

    expansion_re = re.compile(
        r"\b([A-Za-z0-9][A-Za-z0-9+'-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+'-]*){0,5})\s*"
        r"\(([A-Z][A-Z0-9+-]{1,9})\)"
    )
    evidence_lower = all_evidence_text.lower()
    for match in expansion_re.finditer(text):
        expansion, acronym = match.group(1).strip(), match.group(2).strip("+")
        if acronym in _METHOD_ACRONYMS or expansion.lower() in evidence_lower:
            continue
        add(
            f"unverified_identity_{acronym.lower()}",
            "definition_required",
            f"{acronym} may remain a novel chemical proposal, but its expansion is not verified by the packet; provide an unambiguous name plus formula or structure before structure-based claims.",
        )

    acronym_first_re = re.compile(
        r"\b([0-9A-Z][0-9A-Z-]{1,9})(?:\+|\u207a)?\s*\(([^()\n]{4,100})\)"
    )
    for match in acronym_first_re.finditer(text):
        acronym, expansion = match.group(1), match.group(2).strip()
        if acronym in _METHOD_ACRONYMS or expansion.lower() in evidence_lower:
            continue
        add(
            f"unverified_identity_{acronym.lower()}",
            "definition_required",
            f"{acronym} is allowed as a novel proposal, but the packet does not verify this identity. Provide an unambiguous systematic name plus formula, structure, and protonation state before structure-based claims.",
        )

    mentions_3apb = bool(re.search(r"(?i)\b3-APB(?:\+|\u207a)?\b", text))
    mentions_dj = bool(re.search(r"(?i)Dion[- ]Jacobson|\bDJ\b", text))
    has_structural_definition = bool(re.search(
        r"(?i)(?:molecular\s+formula|formula|SMILES|InChI|explicit\s+structure)\s*[:=]", text
    ))
    if mentions_3apb and mentions_dj and not has_structural_definition:
        add(
            "dj_spacer_identity_unresolved",
            "error",
            "The 3-APB shorthand does not establish the bifunctionality, charge, or protonation state required to assign a Dion-Jacobson spacer. Treat the DJ assignment as unverified until a structure is defined.",
        )
    if mentions_3apb and strong_evidence_count == 0 and re.search(
        r"(?i)electronic coupling|charge carrier mobility|trap states?|\u03c0\s*[-–]\s*\u03c0|pi\s*[-–]\s*pi",
        text,
    ):
        add(
            "unverified_3apb_mechanism",
            "boundary",
            "No direct evidence supports the proposed 3-APB coupling, stacking, mobility, or trap mechanism. Present each link as a separate testable inference rather than an established explanation.",
        )

    direct_lower = direct_text.lower()
    quantitative = list(dict.fromkeys(re.findall(
        r"(?i)(?:at\s+least\s+|more\s+than\s+|less\s+than\s+)?\d+(?:\.\d+)?\s*(?:%|percent\b|fold\b)",
        text,
    )))
    unsupported = [item for item in quantitative if item.lower() not in direct_lower]
    if unsupported:
        add(
            "unsupported_quantitative_target",
            "error",
            "Quantitative targets lack property-aligned direct evidence: " + ", ".join(unsupported[:6]) + ". Use directional predictions or label values as exploratory design thresholds.",
        )

    if _near(text, r"\bXRD\b|X-ray diffraction", r"bond(?:ing| strength)?|interfacial interaction"):
        add(
            "xrd_bonding_overreach",
            "method",
            "XRD can measure phase, spacing, texture, and peak widths; it does not directly establish bond strength or interfacial bonding.",
        )
    if _near(text, r"\bSEM\b|scanning electron microscopy", r"bond(?:ing| strength)?|interfacial interaction|defect density"):
        add(
            "sem_mechanism_overreach",
            "method",
            "SEM provides morphology or compositional contrast, not a direct measurement of bonding or electronic defect-state density.",
        )
    if _near(text, r"\bPL\b|photoluminescence|carrier lifetime", r"defect(?:-state)? density"):
        add(
            "pl_defect_density_overreach",
            "method",
            "PL or lifetime changes are recombination observables and cannot alone quantify defect-state density.",
        )
    if _near(text, r"UV[- ]?Vis|absorption spectroscopy", r"\u03c0\s*[-–]\s*\u03c0|pi\s*[-–]\s*pi|stacking interaction"):
        add(
            "uvvis_stacking_overreach",
            "method",
            "UV-Vis band-edge or absorbance changes do not directly establish pi-stacking geometry or interaction strength.",
        )
    if _near(
        text,
        r"\bTAS\b|transient absorption|\bTRPL\b|time-resolved photoluminescence",
        r"\bmobility\b|trap(?:-state)? density",
    ):
        add(
            "spectroscopy_transport_overreach",
            "method",
            "TAS and TRPL probe excited-state and recombination dynamics; they do not alone quantify carrier mobility or trap-state density.",
        )
    if _near(text, r"four[- ]point|electrical conductivity", r"electronic coupling|interfacial coupling"):
        add(
            "conductivity_coupling_overreach",
            "method",
            "Four-point conductivity is a macroscopic transport measurement and does not directly quantify electronic-coupling strength at a 2D/3D interface.",
        )
    if _near(text, r"FWHM|peak width", r"percent crystallinity|crystallinity"):
        add(
            "fwhm_crystallinity_overreach",
            "method",
            "XRD peak width reflects multiple contributions and is not, by itself, a quantitative percent-crystallinity measurement.",
        )
    if _near(text, r"d[- ]?spacing|interlayer spacing", r"bond strength|stronger bond"):
        add(
            "spacing_bond_strength_overreach",
            "method",
            "A change in spacing does not by itself demonstrate stronger bonding; an independent interaction-sensitive measurement is required.",
        )
    if (
        re.search(r"increase\s+in\s+(?:the\s+)?lattice\s+parameter", text, re.IGNORECASE)
        and re.search(r"unit[- ]cell\s+volume[^.]{0,180}decreas", text, re.IGNORECASE)
    ):
        add(
            "lattice_direction_contradiction",
            "internal_consistency",
            "The report predicts both an increased lattice parameter and a decreased unit-cell volume without defining anisotropic axes or a structural model that reconciles them.",
        )
    if _near(text, r"formation energ", r"tighter binding|stronger binding|bond strength"):
        add(
            "formation_energy_binding_overreach",
            "method",
            "Formation energy requires balanced reference states and does not by itself establish local binding strength, especially across different stoichiometries or RP/DJ topologies.",
        )
    if re.search(r"(?is)larger\s+(?:organic\s+)?cations?.{0,240}(?:humidity|water|moisture|stability)", text):
        add(
            "cation_size_confounding",
            "design",
            "A larger-versus-smaller cation comparison also changes chemistry, packing, dimensionality, and energetics. Define a controlled cation series and do not attribute the result to size alone.",
        )
    if _near(text, r"gravimetric|weigh(?:ing)?", r"water permeation|water uptake"):
        add(
            "gravimetric_permeation_controls",
            "design",
            "Thin-film mass gain requires substrate blanks, area/thickness normalization, and phase/decomposition controls before it can be interpreted as permeability.",
        )

    return flags


def render_evidence_contract(
    direct_count: int,
    background_count: int,
    flags: Iterable[dict[str, str]],
) -> str:
    lines = [
        "POLARIS evidence-use contract:",
        "- E# items are directly relevant evidence. B# items are background or analogy only.",
        "- Novel compounds, variables, and mechanisms are allowed. Label them as proposals and define chemical identity unambiguously.",
        "- Every quantitative target needs property-aligned direct evidence or an explicit exploratory-threshold label.",
        "- Match each method to what it directly measures; do not convert a proxy into proof of a mechanism.",
        "- The final evaluator must explicitly resolve every ERROR, METHOD, or DEFINITION_REQUIRED flag and must not call a design robust or well-suited while one remains unresolved.",
        f"- Packet inventory: {direct_count} direct claim(s), {background_count} background claim(s).",
    ]
    quality_flags = list(flags)
    if quality_flags:
        lines.append("Automated pre-analysis checks:")
        for item in quality_flags:
            lines.append(f"- [{item['severity'].upper()}:{item['code']}] {item['message']}")
    return "\n".join(lines)


def bounded_packet_text(blocks: Iterable[str], max_chars: int) -> str:
    """Keep packets within budget without cutting a claim or warning mid-sentence."""
    selected: list[str] = []
    used = 0
    for raw in blocks:
        block = str(raw or "").strip()
        if not block:
            continue
        extra = len(block) + (2 if selected else 0)
        if used + extra > max_chars:
            marker = "[Packet truncated at a block boundary.]"
            if used + len(marker) + 2 <= max_chars:
                selected.append(marker)
            break
        selected.append(block)
        used += extra
    return "\n\n".join(selected)


def guard_hypothesis_output(value: Any) -> Any:
    """Repair deterministic synthesis failures while preserving scientific novelty."""
    if not isinstance(value, str):
        return value
    text = re.sub(r"(?m)^\s*#{2,}\s*$", "", value).strip()
    replacements = (
        (r"(?i)\bincrease\s+by\s+(?:at\s+least\s+|approximately\s+|about\s+)?\d+(?:\.\d+)?\s*%", "increase measurably"),
        (r"(?i)\bimprove\s+by\s+(?:at\s+least\s+|approximately\s+|about\s+)?\d+(?:\.\d+)?\s*%", "improve measurably"),
        (r"(?i)\bdecrease\s+by\s+(?:at\s+least\s+|approximately\s+|about\s+)?\d+(?:\.\d+)?\s*%", "decrease measurably"),
        (r"(?i)\breduction\s+of\s+(?:at\s+least\s+|approximately\s+|about\s+)?\d+(?:\.\d+)?\s*%", "a measurable reduction"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    if (
        re.search(r"(?i)\b3-APB(?:\+|\u207a)?\b", text)
        and re.search(r"(?i)Dion[- ]Jacobson|\bDJ\b", text)
        and "Chemical identity boundary:" not in text
    ):
        boundary = (
            "Chemical identity boundary: 3-APB+ is treated here as a proposed shorthand. "
            "Its systematic identity, formula, explicit structure, protonation state, and compatibility "
            "with a Dion-Jacobson topology must be established before mechanism-based conclusions."
        )
        header = re.match(r"(?is)^\s*(?:\*\*)?Hypothesis\s*:(?:\*\*)?", text)
        if header:
            text = text[:header.end()] + "\n" + boundary + "\n" + text[header.end():].lstrip()
        else:
            text = boundary + "\n\n" + text
    return text.strip()
