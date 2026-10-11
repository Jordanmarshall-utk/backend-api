from __future__ import annotations

import inspect
import re
from functools import wraps
from typing import Any, Callable


_STANDALONE_MARKUP = re.compile(r"(?m)^[ \t]*#{2,}[ \t]*$")
_CHA_INVALID_FORMULA = re.compile(
    r"CHA\s*(?:\+|\u207a)?\s*\(\s*PbI(?:4|\u2084)\s*\)\s*(?:2|\u2082)",
    re.IGNORECASE,
)
_CHA_RP_FORMULA = "(CHA)\u2082PbI\u2084"
_CHA_RP_FORMULA_PATTERN = re.compile(
    r"\(\s*CHA\s*\)\s*(?:2|\u2082)\s*PbI(?:4|\u2084)", re.IGNORECASE
)
_PERCENT = r"(?:\d+(?:\.\d+)?\s*(?:-|\u2013|to)\s*)?\d+(?:\.\d+)?\s*%"
_PREDICTED_ENERGY = re.compile(
    r"\b(estimated|predicted|anticipated|expected|targeted)"
    r"(\s+to\s+be)?(\s+(?:at|around|approximately|about|roughly))?"
    r"\s+\d+(?:\.\d+)?\s*(?:meV|eV)\b",
    re.IGNORECASE,
)
_DIAMMONIUM_DJ_SCOPE = re.compile(
    r"(?is)(?=.*\bdiammonium\b)(?=.*(?:\bDJ\b|Dion[- ]Jacobson))"
)
_NMEA_MONOCATION = re.compile(
    r"\bN-methyl-2-aminoethanol\b|\bNMEA(?:\+|\u207a)?\b", re.IGNORECASE
)
_PHENETHYLAMMONIUM_MONOCATION = re.compile(
    r"\b4-fluoro(?:phenethylammonium|phenethylamine)\b|\b4-FPEA(?:\+|\u207a)?\b|"
    r"\bphenethylammonium\b|\bPEA(?:\+|\u207a)?\b",
    re.IGNORECASE,
)


def _dj_candidate_class_boundary(text: str) -> str | None:
    """Return a hard boundary for known monocations assigned to a diammonium DJ class."""
    if not _DIAMMONIUM_DJ_SCOPE.search(str(text or "")):
        return None
    if _NMEA_MONOCATION.search(text):
        identity = "N-methyl-2-aminoethanol/NMEA has one protonatable amine"
    elif _PHENETHYLAMMONIUM_MONOCATION.search(text):
        identity = "Phenethylammonium and 4-fluorophenethylammonium are monovalent spacers"
    else:
        return None
    return (
        f"{identity} and therefore does not satisfy the requested diammonium DJ class. "
        "It may be retained only as an explicitly monovalent RP/passivation comparator. "
        "A DJ candidate requires two verified protonatable amines, an intended diprotonated "
        "2+ state, and an explicit structure or SMILES."
    )


def clean_model_markup(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = _STANDALONE_MARKUP.sub("", value)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if isinstance(value, list):
        return [clean_model_markup(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clean_model_markup(item) for item in value)
    if isinstance(value, dict):
        return {key: clean_model_markup(item) for key, item in value.items()}
    return value


def strip_unsupported_quantitative_targets(text: str) -> tuple[str, bool]:
    original = text
    text = re.sub(
        rf"\bat\s+least\s+{_PERCENT}\s+(greater|higher|lower|smaller|faster|slower)\b",
        lambda match: f"measurably {match.group(1)}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:an?\s+)?{_PERCENT}\s+(reduction|increase|decrease|improvement)\b",
        lambda match: f"a measurable {match.group(1)}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(reduction|increase|decrease|improvement)\s+of\s+{_PERCENT}\b",
        lambda match: f"measurable {match.group(1)}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(reduce|decrease|increase|improve|enhance)(d|s|ing)?\s+by\s+(?:at\s+least\s+)?{_PERCENT}\b",
        lambda match: f"{match.group(1)}{match.group(2) or ''} measurably",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:approximately|about|around|roughly|nearly|up\s+to)\s+{_PERCENT}(?!\w)",
        "a measurable but presently uncalibrated amount",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b{_PERCENT}(?!\w)",
        "a measurable but presently uncalibrated amount",
        text,
        flags=re.IGNORECASE,
    )
    text = _PREDICTED_ENERGY.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2) or ''} "
            "an empirically determined magnitude"
        ),
        text,
    )
    return text, text != original


def _three_apb_boundary(text: str) -> str | None:
    if not re.search(r"\b3[- ]APB(?:\+|\u207a)?\b", text, re.IGNORECASE):
        return None
    checks = {
        "molecular formula": bool(re.search(r"\bC\d+H\d+(?:N\d*)?(?:\+|\u207a|2\+)?\b", text)),
        "unambiguous structure": bool(re.search(r"\b(?:SMILES|InChI|structural formula)\b", text, re.IGNORECASE)),
        "protonation state and net charge": bool(
            re.search(r"\bprotonation state\b", text, re.IGNORECASE)
            and re.search(r"\b(?:net charge|charge state|monocation|dication|1\+|2\+)\b", text, re.IGNORECASE)
        ),
        "counterion": bool(re.search(r"\bcounterion\b", text, re.IGNORECASE)),
    }
    missing = [name for name, present in checks.items() if not present]
    if not missing:
        return None
    boundary = (
        "3-APB+ may remain a novel proposal, but structure-based mechanisms cannot be evaluated "
        f"until the following are supplied: {', '.join(missing)}."
    )
    if re.search(r"\b(?:DJ|Dion[- ]Jacobson)\b", text, re.IGNORECASE):
        boundary += (
            " A DJ assignment additionally requires confirmation that the intended spacer has two "
            "protonatable sites and a net 2+ charge under the stated processing conditions."
        )
    return boundary


def _composition_gate(text: str) -> tuple[str, str | None]:
    if not _CHA_INVALID_FORMULA.search(text):
        return text, None
    unresolved = "the proposed CHA-containing layered phase (stoichiometry unresolved)"
    guarded = _CHA_INVALID_FORMULA.sub(unresolved, text)
    return guarded, (
        "The generated CHA+(PbI4)2 notation is chemically ambiguous and is not charge-balanced "
        "as written. CHA must not be assigned to a specific molecule without a supplied systematic "
        "name, molecular formula, unambiguous structure, protonation state, and net charge. If the "
        f"intended cation is confirmed to be monovalent, {_CHA_RP_FORMULA} is one candidate n=1 RP "
        "stoichiometry, not an established identity or phase assignment."
    )


def _cha_identity_boundary(text: str) -> str | None:
    if not re.search(r"\bCHA\s*(?:\+|\u207a)?\b", text, re.IGNORECASE):
        return None
    checks = {
        "systematic name": bool(
            re.search(r"\b(?:IUPAC|systematic name)\b", text, re.IGNORECASE)
        ),
        "molecular formula": bool(
            re.search(r"\bC\d+H\d+(?:N\d*)?(?:\+|\u207a|2\+)?\b", text)
        ),
        "unambiguous structure": bool(
            re.search(r"\b(?:SMILES|InChI|structural formula)\b", text, re.IGNORECASE)
        ),
        "protonation state and net charge": bool(
            re.search(r"\bprotonation state\b", text, re.IGNORECASE)
            and re.search(
                r"\b(?:net charge|charge state|monocation|dication|1\+|2\+)\b",
                text,
                re.IGNORECASE,
            )
        ),
    }
    missing = [name for name, present in checks.items() if not present]
    if not missing:
        return None
    return (
        "CHA may remain a novel candidate label, but its identity must remain unresolved until "
        f"the following are supplied by the user or a cited source: {', '.join(missing)}. "
        "No specific molecular identity, ring structure, stoichiometry, or structure-derived "
        "mechanism may be inferred from the abbreviation alone."
    )


def _remove_unsupported_cha_assignments(text: str) -> str:
    guarded = _CHA_RP_FORMULA_PATTERN.sub(
        "the proposed CHA-containing layered phase (stoichiometry unresolved)", text
    )
    guarded = re.sub(
        r"\bcyclohexylammonium\b", "proposed CHA species", guarded, flags=re.IGNORECASE
    )
    guarded = re.sub(
        r"\bcyclohexyl ring\b", "proposed organic moiety", guarded, flags=re.IGNORECASE
    )
    guarded = re.sub(
        r"\bC6H14N(?:\+|\u207a)?\b", "formula unresolved", guarded, flags=re.IGNORECASE
    )
    guarded = re.sub(
        r"(chemical identity of CHA[^.]{0,160}?)\bis defined\b",
        r"\1remains unresolved",
        guarded,
        flags=re.IGNORECASE,
    )
    return guarded


def _guard_transport_and_outcome_claims(text: str) -> tuple[str, list[str]]:
    guarded = text
    boundaries: list[str] = []

    activation_pattern = re.compile(
        r"\b(reduce|reduces|reduced|reducing|lower|lowers|lowered|lowering)\s+"
        r"(?:the\s+)?(?:(?:activation\s+)?energy(?:\s+required)?|activation\s+barrier)\s+"
        r"for\s+ion\s+migration\b",
        flags=re.IGNORECASE,
    )
    if activation_pattern.search(guarded):
        verb_map = {
            "reduce": "increase",
            "reduces": "increases",
            "reduced": "increased",
            "reducing": "increasing",
            "lower": "increase",
            "lowers": "increases",
            "lowered": "increased",
            "lowering": "increasing",
        }

        def replace_activation(match: re.Match) -> str:
            verb = verb_map[match.group(1).lower()]
            return f"{verb} the effective activation barrier for ion migration"

        guarded = activation_pattern.sub(replace_activation, guarded)
        boundaries.append(
            "Suppressing a thermally activated migration pathway requires an increased effective "
            "activation barrier; lowering that barrier would facilitate migration."
        )

    trpl = r"(?:time-resolved\s+photoluminescence(?:\s+decay)?(?:\s+experiments?)?|TRPL)"
    trpl_direct = bool(
        re.search(
            rf"(?:{trpl})[^.]{{0,180}}(?:quantif\w*|measur\w*|reflect\w*)[^.]{{0,80}}"
            r"(?:ion\s+migration|migration\s+rates?)",
            guarded,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"(?:ion\s+migration|migration\s+rates?)[^.]{{0,180}}(?:measured\s+by|using)\s+{trpl}",
            guarded,
            flags=re.IGNORECASE,
        )
    )
    if trpl_direct:
        guarded = re.sub(
            rf"{trpl}[^.;]{{0,140}}(?:quantif\w*|measur\w*|determin\w*)"
            r"[^.;]{0,60}(?:ion\s+migration(?:\s+rates?)?|migration\s+rates?)",
            "dedicated bias-, time-, and temperature-dependent ion-transport measurements "
            "will quantify ion migration; time-resolved photoluminescence will separately "
            "characterize carrier recombination",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            rf"{trpl}\s+(?:will|can)\s+(?:be\s+conducted\s+to\s+)?"
            r"(?:quantify|measure|determine)\s+(?:the\s+)?(?:ion\s+migration|migration\s+rates?)",
            "dedicated bias- and temperature-dependent ion-transport measurements will quantify ion migration; "
            "time-resolved photoluminescence will separately characterize carrier recombination",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            rf"(?P<prefix>(?:ion\s+migration|migration\s+rates?)[^.]{{0,180}}?)"
            rf"(?:as\s+measured\s+by|using)\s+{trpl}",
            r"\g<prefix>using dedicated bias- and temperature-dependent ion-transport measurements; "
            "time-resolved photoluminescence may separately assess carrier recombination",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:the\s+)?decay\s+rate\s+constant\s*,?\s*which\s+reflects\s+"
            r"(?:the\s+)?rate\s+of\s+ion\s+migration",
            "the photoluminescence decay constant, which reflects carrier recombination dynamics "
            "rather than an ion-migration rate",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "Time-resolved photoluminescence measures carrier recombination dynamics, not an ion-migration "
            "rate. Ion transport requires dedicated bias-, time-, and temperature-dependent measurements."
        )

    grain_boundary_claim = bool(
        re.search(
            r"(?:XRD[^.]{0,260}SEM|SEM[^.]{0,260}XRD)[^.]{0,260}grain[- ]boundary\s+density",
            guarded,
            flags=re.IGNORECASE,
        )
    )
    if grain_boundary_claim:
        guarded = re.sub(
            r"(?:XRD\s+and\s+SEM|SEM\s+and\s+XRD)[^.]{0,320}grain[- ]boundary\s+density[^.]*\.",
            "XRD can assess phase and average crystallinity, while SEM can assess surface morphology "
            "and apparent grain size. Grain-boundary density requires dedicated cross-sectional or "
            "three-dimensional microscopy.",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "XRD and surface SEM do not directly quantify grain-boundary density; use dedicated "
            "cross-sectional or three-dimensional microscopy for that claim."
        )

    lower = guarded.lower()
    pce_overclaim = bool(
        ("pce" in lower or "power conversion efficiency" in lower)
        and ("ion migration" in lower or _states_no_direct_support(guarded))
        and re.search(
            r"\b(?:projected\s+increase|expected\s+increase|"
            r"expected\s+to\s+translate\s+into\s+improved|"
            r"expected\s+to\s+yield\s+a\s+higher\s+pce|"
            r"predicted\s+to\s+yield\s+a\s+higher\s+pce|"
            r"(?:pce|power\s+conversion\s+efficiency)[^.]{0,100}will\s+increase|"
            r"(?:pce|power\s+conversion\s+efficiency)[^.]{0,100}should\s+be\s+higher|"
            r"(?:an?\s+)?increase\s+in\s+pce[^.]{0,100}would\s+support|"
            r"translate\s+into\s+improved\s+long-term\s+device\s+performance)\b",
            lower,
        )
    )
    if pce_overclaim:
        def replace_expected_pce(match: re.Match) -> str:
            agreement = "does" if match.group(1).lower() == "is" else "do"
            return (
                f"{agreement} not establish a direction for PCE, which should be evaluated "
                "separately as a secondary device-level outcome"
            )

        guarded = re.sub(
            r"\b(is|are)\s+expected\s+to\s+translate\s+into\s+(?:an?\s+)?improved\s+"
            r"(?:PCE|power\s+conversion\s+efficiency(?:\s*\(PCE\))?)",
            replace_expected_pce,
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:is|are)\s+(?:also\s+)?expected\s+to\s+translate\s+into\s+improved\s+"
            r"long-term\s+device\s+performance",
            "must be evaluated separately as a device-level outcome",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:a\s+)?projected\s+increase\s+in\s+power\s+conversion\s+efficiency\s*\(PCE\)"
            r"(?:\s+of\s+[^,.;]+)?",
            "power conversion efficiency (PCE) reported as a secondary outcome without a prespecified direction",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:is|are)\s+(?:expected|predicted|anticipated)\s+to\s+yield\s+a\s+higher\s+PCE\s+"
            r"compared\s+to\s+([^.]+)",
            r"will have PCE reported as a secondary device-level outcome without a prespecified "
            r"direction relative to \1",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?i)(?:the\s+)?(?:power\s+conversion\s+efficiency\s*\(PCE\)|PCE)\s+of\s+"
            r"[^.]{0,180}?\s+will\s+increase(?:\s+measurably)?[^.]*\.",
            "Power conversion efficiency (PCE) will be reported as a secondary device-level "
            "outcome without a prespecified direction; it will not validate the proposed "
            "ion-transport mechanism.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)(?:the\s+key\s+parameter\s+is\s+)?(?:the\s+)?PCE\s+value,?\s+which\s+"
            r"should\s+be\s+higher[^.]*\.",
            "PCE will be compared as a secondary outcome without a prespecified direction and "
            "cannot confirm the proposed mechanism.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)(?:an?\s+)?increase\s+in\s+PCE\s+would\s+support[^.]*\.",
            "A PCE difference would be a secondary device-level observation and would not by "
            "itself support the proposed ion-transport mechanism.",
            guarded,
        )
        boundaries.append(
            "PCE is a system-level secondary outcome and cannot by itself validate an ion-migration "
            "mechanism or be assigned a direction without supporting evidence."
        )

    return guarded, boundaries


def _states_no_direct_support(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:no\s+direct(?:ly relevant)?\s+(?:empirical\s+)?evidence|"
            r"no\s+direct\s+support|0\s+direct\s+claim|grounding\s+unavailable)",
            str(text or ""),
        )
    )


def _strip_citations_without_direct_support(text: str) -> str:
    cleaned_lines: list[str] = []
    for original_line in str(text or "").splitlines():
        line = original_line
        anchor = re.search(r"(?i)\**Evidence\s+anchor\s*:\**", line)
        if anchor:
            before = line[:anchor.start()].rstrip()
            boundary = re.search(r"(?i)\**Evidence\s+boundary\s*:\**", line[anchor.end():])
            after = ""
            if boundary:
                boundary_start = anchor.end() + boundary.start()
                after = line[boundary_start:].strip()
            line = " ".join(part for part in (before, after) if part)
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [
            sentence
            for sentence in sentences
            if not re.search(r"\[(?:E|B|KG)\d+\]|\bDOI\s*:\s*10\.", sentence, re.IGNORECASE)
        ]
        line = " ".join(kept).strip()
        line = re.sub(
            r"\(\s*(?:E|B|KG)\d+(?:\s*(?:,|and|through|-)\s*(?:E|B|KG)\d+)*\s*\)",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\[(?:E|B|KG)\d+\]", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\b(?:E|B|KG)\d+\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\(?\s*DOI\s*:\s*10\.\S+\)?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\(\s*(?:and|,|-)?\s*\)", "", line)
        line = re.sub(r"[ \t]{2,}", " ", line)
        cleaned_lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def _strip_internal_process_notes(text: str) -> str:
    cleaned = re.sub(
        r"(?ims)^\s*---\s*\n+\s*\*\*Note on (?:Kantian|Humean|philosophical|audit)[^\n]*\*\*"
        r"[^\n]*(?:\n(?!\s*(?:#{1,6}\s|\*\*(?:Calibrated status|Automated quality control))))*",
        "",
        str(text or ""),
    )
    cleaned = re.sub(
        r"(?im)^\s*\*\*Note on (?:Kantian|Humean|philosophical|audit)[^\n]*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^.*\b(?:report adheres to|Kantian constraint(?:s|\s+checking)?|"
        r"Humean constraint(?:s|\s+checking)?|philosophy manager)\b.*$",
        "",
        cleaned,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _guard_spacer_and_method_claims(text: str) -> tuple[str, list[str]]:
    guarded = str(text or "")
    boundaries: list[str] = []
    eda_token = r"(?<![A-Za-z0-9])EDA(?:2\+|\u00b2\u207a)?(?![A-Za-z0-9])"
    dap_token = r"(?<![A-Za-z0-9])DAP(?:2\+|\u00b2\u207a)?(?![A-Za-z0-9])"
    pea_token = r"(?<![A-Za-z0-9])PEA(?:\+|\u207a)?(?![A-Za-z0-9])"
    has_eda = bool(re.search(rf"{eda_token}|ethylenediammonium", guarded, re.IGNORECASE))
    has_dap = bool(re.search(rf"{dap_token}|1,3-diaminopropane|propan(?:e-1,3|e-1,3)-diammonium", guarded, re.IGNORECASE))
    has_pea = bool(re.search(rf"{pea_token}|phenethylammonium", guarded, re.IGNORECASE))

    if has_eda:
        replacements = (
            (r"stronger electrostatic interactions and coordination capabilities", "a proposed difference in ionic and hydrogen-bonding interactions"),
            (r"stronger electrostatic interactions", "different ionic interactions"),
            (r"coordination capabilities", "proposed hydrogen-bonding geometry"),
            (r"general principles of electrostatics and coordination chemistry", "general principles of ionic and hydrogen-bonding interactions"),
            (r"stronger binding and bridging capabilities", "proposed bifunctional slab-linking geometry"),
            (r"more stable and extensive bridges", "a proposed slab-linking geometry"),
            (r"denser network of bonds", "different interfacial interaction network"),
            (r"denser and more stable network", "different local interaction topology"),
            (r"more robust barrier against ion movement", "different barrier landscape for ion movement"),
            (r"enhanced Coulombic interactions", "a proposed difference in Coulombic interactions"),
            (r"dense network formed by EDA(?:2\+|\u00b2\u207a)?", "interfacial structure formed with EDA2+"),
        )
        for pattern, replacement in replacements:
            guarded = re.sub(pattern, replacement, guarded, flags=re.IGNORECASE)
        identity = (
            "EDA2+ is assumed to mean ethane-1,2-diammonium, the diprotonated form of "
            "ethane-1,2-diamine (C2H10N2^2+). The counterion, phase stoichiometry, and "
            "protonation state under processing conditions must be specified."
        )
        if has_pea:
            identity += (
                " PEA+ is assumed to mean phenethylammonium (C8H12N+). Comparing EDA2+ "
                "with PEA+ changes charge, ammonium-site count, linker length, aromaticity, "
                "flexibility, and hydrophobicity, so that comparison cannot isolate valence alone."
            )
        elif re.search(r"\bmonovalent\b|\bdivalent nature\b", guarded, re.IGNORECASE):
            identity += (
                " Comparing EDA2+ with an unspecified monovalent spacer changes valence together "
                "with molecular structure, ammonium-site count, size, flexibility, and interfacial "
                "chemistry, so that comparison cannot isolate valence alone. A controlled spacer "
                "series or factorial design is required for a valence-specific conclusion."
            )
        boundaries.append(identity)

    if has_dap:
        guarded = re.sub(
            r"1,3-diaminopropane\s+dication",
            "proposed propane-1,3-diammonium dication",
            guarded,
            flags=re.IGNORECASE,
        )
        replacements = (
            (r"stronger electrostatic interactions?", "a proposed difference in electrostatic interactions"),
            (r"enhanced electrostatic interactions?", "a proposed difference in electrostatic interactions"),
            (r"tighter binding", "different interfacial binding geometry"),
            (r"stronger binding", "different interfacial binding geometry"),
            (r"greater structural stability", "a potentially different structural response"),
            (r"well-aligned electronic bands", "electronic alignment that remains to be measured"),
        )
        for pattern, replacement in replacements:
            guarded = re.sub(pattern, replacement, guarded, flags=re.IGNORECASE)
        identity = (
            "DAP2+ is provisionally assumed to mean propane-1,3-diammonium, the diprotonated "
            "form of propane-1,3-diamine (C3H12N2^2+). Because DAP is an ambiguous acronym, "
            "the systematic name, structure, counterion, phase stoichiometry, and protonation "
            "state under processing conditions must be confirmed."
        )
        if has_pea:
            identity += (
                " PEA+ is assumed to mean phenethylammonium (C8H12N+). Comparing DAP2+ with "
                "PEA+ changes charge, ammonium-site count, linker length, aromaticity, flexibility, "
                "and hydrophobicity, so that comparison cannot isolate valence alone. A controlled "
                "spacer series or factorial design is required for a valence-specific conclusion."
            )
        elif re.search(r"\bmonovalent\b|\bdouble positive charge\b|\bdivalent\b", guarded, re.IGNORECASE):
            identity += (
                " Comparing DAP2+ with an unspecified monovalent spacer changes valence together "
                "with molecular structure, ammonium-site count, size, flexibility, and interfacial "
                "chemistry, so that comparison cannot isolate valence alone. A controlled spacer "
                "series or factorial design is required for a valence-specific conclusion."
            )
        boundaries.append(identity)

    if has_pea and not has_eda and not has_dap:
        guarded = re.sub(
            r"novel\s+chemical\s+species\s*\(PEA(?:\+|\u207a)?\)",
            "established phenethylammonium spacer chemistry used in a newly proposed comparison",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:the\s+)?linear\s+(?:alkyl\s+chain|structure)\s+of\s+PEA(?:\+|\u207a)?|"
            r"PEA(?:\+|\u207a)?['\u2019]s\s+linear\s+structure",
            "the phenyl-ethyl-ammonium structure of PEA+",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"\blinear\s+structure\b",
            "phenyl-ethyl-ammonium structure",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"less\s+steric\s+hindrance\s+compared\s+to\s+bulkier\s+divalent\s+spacers",
            "a different steric profile from an as-yet unspecified divalent comparator",
            guarded,
            flags=re.IGNORECASE,
        )
        identity = (
            "PEA+ is assumed to mean phenethylammonium (C8H12N+), the protonated form of "
            "2-phenylethan-1-amine. The counterion, phase stoichiometry, and protonation state "
            "under processing conditions must be specified."
        )
        if re.search(r"\bdivalent\b|\bdication\b", guarded, re.IGNORECASE):
            identity += (
                " Comparing PEA+ with an unspecified divalent spacer changes valence together with "
                "molecular structure, ammonium-site count, size, flexibility, aromaticity, and "
                "interfacial chemistry, so that comparison cannot isolate valence alone. A controlled "
                "spacer series or factorial design is required for a valence-specific conclusion."
            )
        boundaries.append(identity)

    lower = guarded.lower()
    if "tof-sims" in lower or "tofsims" in lower:
        guarded = re.sub(
            r"ToF-SIMS\s+(?:will|would|can)\s+(?:be\s+used\s+to\s+)?(?:directly\s+)?"
            r"(?:measure|quantify|determine)\s+(?:the\s+)?"
            r"(?:ion\s+migration(?:\s+rates?)?|migration\s+rates?)",
            "ToF-SIMS will map stress-dependent ion redistribution; migration kinetics require "
            "controlled bias-, time-, and temperature-dependent measurements",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"ToF-SIMS\s+to\s+measure\s+the\s+depth\s+profile\s+of\s+migrating\s+ions",
            "ToF-SIMS to map ion distributions after controlled bias-, time-, and temperature-dependent stressing",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"the\s+concentration\s+of\s+migrating\s+ions\s+at\s+various\s+depths",
            "stress-dependent ion-depth profiles at predefined times",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"provide\s+quantitative\s+data\s+on\s+the\s+effectiveness\s+of\s+the\s+spacers\s+in\s+suppressing\s+ion\s+migration",
            "test whether ion redistribution differs between matched devices; ToF-SIMS alone does not yield a migration rate",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"\(?ToF-SIMS\)?,?\s+which\s+measures?\s+the\s+movement\s+of\s+ions\s+through\s+"
            r"the\s+perovskite\s+structure\s+over\s+time",
            "ToF-SIMS, which maps ion distributions after predefined electrical, thermal, and temporal stress",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"Ion\s+migration\s+rates?\s+will\s+be\s+assessed\s+using\s+"
            r"time-of-flight\s+secondary\s+ion\s+mass\s+spectrometry\s*\(ToF-SIMS\)",
            "Stress-induced ion redistribution will be mapped using time-of-flight secondary ion "
            "mass spectrometry (ToF-SIMS)",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "ToF-SIMS maps stress-dependent ion distributions. Migration kinetics require a controlled "
            "bias/time/temperature series plus a complementary transient-current or impedance measurement."
        )

    if "trmc" in lower or "time-resolved microwave conductivity" in lower:
        guarded = re.sub(
            r"TRMC\s+(?:will|would|can)\s+(?:directly\s+)?"
            r"(?:measure|assess|determine)\s+(?:(?:charge[- ]carrier|carrier)\s+)?mobility",
            "TRMC will measure transient photoconductivity, which depends on photogeneration "
            "yield times the sum of carrier mobilities",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"Charge\s+carrier\s+mobility\s+will\s+be\s+assessed\s+using\s+time-resolved\s+microwave\s+conductivity\s*\(TRMC\)\s+measurements?",
            "TRMC will assess transient photoconductivity, proportional to photogeneration yield times the sum of carrier mobilities",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"The\s+mobility\s+values\s+obtained\s+for",
            "The TRMC photoconductivity responses obtained for",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"Lower\s+mobility\s+in\s+EDA(?:2\+|\u00b2\u207a)?\s+devices\s+would\s+indicate[^.]*\.",
            "A lower TRMC response alone would not distinguish changes in photogeneration yield, mobility, trapping, or recombination.",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "TRMC reports a transient photoconductivity observable, commonly expressed as photogeneration "
            "yield times summed mobility; it is not an unqualified standalone mobility measurement."
        )

    if "impedance spectroscopy" in lower or re.search(r"\bEIS\b", guarded):
        guarded = re.sub(
            r"(?:EIS|(?:electrochemical\s+)?impedance\s+spectroscopy(?:\s*\(EIS\))?)\s+"
            r"(?:will|would|can)\s+(?:be\s+)?(?:(?:used|employed)\s+)?(?:to\s+)?measure\s+"
            r"(?:the\s+)?ion\s+migration\s+rates?",
            "Temperature- and bias-dependent EIS will test mixed transport-response differences; "
            "migration kinetics require validated deconvolution and a complementary time-resolved ion measurement",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:Utilize|Use)\s+electrochemical impedance spectroscopy\s*\(EIS\)\s+to\s+"
            r"measure\s+(?:the\s+)?ion(?:ic)?\s+conductivity",
            "Use temperature- and bias-dependent electrochemical impedance spectroscopy (EIS), "
            "with ion-blocking and nonblocking controls, to estimate ionic contributions to the response",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"This\s+test\s+is\s+(?:crucial|essential)\s+as\s+it\s+directly\s+measures?\s+"
            r"(?:the\s+)?impact[^.]{0,160}ion\s+dynamics[^.]*\.",
            "With the stated controls, EIS can test whether the mixed electrical response changes "
            "but cannot directly assign ion dynamics.",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"The\s+key\s+parameter\s+(?:here\s+)?is\s+the\s+resistance\s+associated\s+with\s+"
            r"ion\s+migration,?\s+which\s+should\s+be[^.]*\.",
            "Equivalent-circuit parameters will be compared only after their ionic, electronic, "
            "dielectric, and contact contributions are independently constrained.",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"as\s+measured\s+by\s+electrochemical\s+impedance\s+spectroscopy\s*\(EIS\)",
            "tested using controlled EIS together with a complementary time-resolved ion measurement",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"EIS\s+is\s+a\s+valid\s+method\s+for\s+measuring\s+ion(?:ic)?\s+conductivity",
            "EIS can help estimate ionic contributions only after an appropriate equivalent-circuit "
            "model is validated against electrode and electronic-transport controls",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:The\s+)?ion\s+conductivity\s+measurements?\s+will\s+directly\s+assess\s+"
            r"(?:the\s+)?impact\s+on\s+ion\s+movement",
            "The impedance response will test for transport differences but will not by itself "
            "assign an ion-migration mechanism",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"This\s+test\s+will\s+provide\s+quantitative\s+evidence\s+of\s+the\s+impact\s+of\s+"
            rf"{eda_token}\s+on\s+ion\s+migration",
            "This test can estimate whether the frequency- and temperature-dependent transport "
            "response changes, subject to the stated deconvolution controls",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"Evaluates?\s+ion\s+migration\s+rates?\s+by\s+analyzing\s+electrical\s+resistance",
            "Measures a frequency-dependent mixed electrical response that requires ionic/electronic "
            "deconvolution and cannot alone yield an ion-migration rate",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "EIS responses can mix ionic and electronic transport, dielectric polarization, and "
            "contact effects. Assigning ionic conductivity or migration kinetics requires validated "
            "equivalent-circuit analysis, blocking/nonblocking controls, temperature and bias series, "
            "and a complementary time-resolved ion-redistribution measurement."
        )

    pl_defect_claim = bool(
        re.search(r"\b(?:PL|photoluminescence)\b", guarded, re.IGNORECASE)
        and re.search(r"defect(?:-state)?\s+density|fewer\s+defects", guarded, re.IGNORECASE)
    )
    if pl_defect_claim:
        guarded = re.sub(
            r"\*\*Defect Density Analysis:\*\*",
            "**Recombination and defect-sensitive proxy analysis:**",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:Employ|Use)\s+photoluminescence spectroscopy\s*(?:\(PL\))?\s+to\s+"
            r"assess\s+(?:the\s+)?defect(?:-state)?\s+density",
            "Use steady-state and time-resolved photoluminescence to characterize recombination, "
            "paired with an independent trap-sensitive measurement",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:with\s+)?a\s+higher\s+(?:PL\s+)?signal\s+indicating\s+fewer\s+defects",
            "without assigning PL intensity alone to a unique defect concentration",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"Develop\s+a\s+calibration\s+method\s+for\s+correlating\s+changes\s+in\s+PL\s+"
            r"intensity\s+and\s+lifetime\s+with\s+defect\s+density",
            "Predefine how PL observables will be interpreted alongside an independent trap-sensitive "
            "measurement rather than converted directly into defect density",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:PL|photoluminescence(?:\s+spectroscopy)?)\s+can\s+estimate\s+defect\s+density",
            "PL can characterize recombination changes but cannot uniquely estimate defect-state density",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:Assess|Estimate)\s+(?:the\s+)?defect\s+density,?\s*(?:focusing\s+on|using)\s+"
            r"(?:the\s+)?PL\s+intensity\s+and\s+lifetime",
            "Characterize PL intensity and lifetime as recombination observables",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"Higher\s+values\s+suggest\s+fewer\s+defects",
            "Neither higher intensity nor longer lifetime uniquely identifies a lower defect concentration",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"a\s+reduction\s+in\s+defects\s+would\s+correspond\s+to\s+an?\s+increase\s+in\s+"
            r"photoluminescence\s+intensity",
            "PL changes would require interpretation against optical, compositional, and trap-sensitive controls",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "PL intensity and lifetime are recombination observables and cannot alone quantify "
            "defect-state density; composition, absorption, excitation density, photon outcoupling, "
            "and an independent trap-sensitive measurement must be controlled."
        )

    guarded = re.sub(
        r"(?i)(photoluminescence\s*(?:\(PL\))?\s*(?:spectroscopy)?\s*:?\s*)"
        r"(?:will\s+be\s+(?:used|utilized)\s+to\s+)?(?:evaluate|measure|assess|quantify)\s+"
        r"(?:the\s+)?defect(?:-state)?\s+density[^.]*\.",
        r"\1will characterize recombination dynamics; defect-state density requires a separate "
        r"trap-sensitive measurement with optical and compositional controls.",
        guarded,
    )
    guarded = re.sub(
        r"(?i)(?:the\s+prediction\s+is\s+)?a\s+reduction\s+in\s+defect(?:-state)?\s+density,?\s+"
        r"which\s+may\s+be\s+indicated\s+by\s+changes\s+in\s+PL\s+intensity\s+and\s+lifetime\.",
        "Changes in PL intensity and lifetime would be interpreted as recombination observables, "
        "not as a unique defect-density measurement.",
        guarded,
    )
    guarded = re.sub(
        r"(?i)(?:the\s+)?defect(?:-state)?\s+density,?\s+as\s+measured\s+by\s+"
        r"photoluminescence(?:\s+spectroscopy)?,?\s+would\s+be\s+reduced[^.]*\.",
        "An independent trap-sensitive measurement would test for a change in defect-state "
        "density, while PL would separately characterize recombination dynamics.",
        guarded,
    )

    eda_alias_claim = bool(
        re.search(r"\bethylenediamine\s+dication\b", guarded, re.IGNORECASE)
        or re.search(r"\bED(?=2\+|\u00b2\u207a)", guarded, re.IGNORECASE)
    )
    if eda_alias_claim:
        guarded = re.sub(
            r"\bED(?=2\+|\u00b2\u207a)",
            "EDA",
            guarded,
            flags=re.IGNORECASE,
        )
        eda_replacements = (
            (r"\btighter binding\b", "a testable difference in local interaction geometry"),
            (r"\bstronger binding\b", "a testable difference in local interaction geometry"),
            (r"\bstronger electrostatic interactions?\b", "different electrostatic interaction geometry and energetics"),
            (r"\benhanced electrostatic interactions?\b", "different electrostatic interaction geometry and energetics"),
            (r"\bhigher charge density\b", "different charge distribution and bifunctional geometry"),
            (r"\bmultiple coordination sites within the lattice\b", "multiple candidate hydrogen-bonding contacts at opposing inorganic slabs"),
        )
        for pattern, replacement in eda_replacements:
            guarded = re.sub(pattern, replacement, guarded, flags=re.IGNORECASE)
        guarded = re.sub(
            r"(?i)result\s+in\s+a\s+measurable\s+increase\s+in\s+the\s+lattice\s+parameter,?\s+"
            r"indicative\s+of\s+[^.]*\.",
            "produce a measurable change in refined lattice parameters; lattice dimensions alone "
            "would not establish interaction strength.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)the\s+unit\s+cell\s+volume[^.]{0,260}(?:predicted|expected)\s+to\s+decrease[^.]*\.",
            "Refined unit-cell metrics are predicted to differ, but their direction should remain "
            "uncalibrated until phase, composition, strain, temperature, and topology are matched.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)(?:a\s+)?lower\s+formation\s+energy[^.]{0,260}(?:supporting|support)\s+"
            r"(?:the\s+notion\s+of\s+)?[^.]*\.",
            "Formation energies computed against an explicitly balanced, shared set of reference "
            "states would test relative thermodynamic stability; they would not alone establish "
            "local interaction strength.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)(SEM\s+and\s+TEM|scanning\s+electron\s+microscopy\s*\(SEM\)\s+and\s+"
            r"transmission\s+electron\s+microscopy\s*\(TEM\))[^.]{0,360}providing\s+direct\s+"
            r"evidence[^.]*\.",
            "SEM can assess morphology, and appropriately calibrated TEM or electron diffraction "
            "can probe local structure; neither alone establishes interaction strength.",
            guarded,
        )
        boundaries.append(
            "EDA2+ denotes doubly protonated ethylenediamine (ethane-1,2-diaminium; C2H10N2^2+), "
            "not neutral ethylenediamine; protonation state, counterions, stoichiometry, and the "
            "resulting dimensional phase must be specified."
        )
        if re.search(r"\bmonovalent\b", guarded, re.IGNORECASE):
            boundaries.append(
                "Comparing EDA2+ with a monovalent spacer changes charge, molecular geometry, "
                "stoichiometry, and RP/DJ topology simultaneously, so it cannot isolate valence "
                "alone; use a chemically controlled series or factorial design."
            )

    trpl_proxy_claim = bool(
        re.search(r"\bTRPL\b|time-resolved\s+photoluminescence", guarded, re.IGNORECASE)
        and re.search(r"trap\s+density|charge\s+carrier\s+mobility", guarded, re.IGNORECASE)
    )
    if trpl_proxy_claim:
        guarded = re.sub(
            r"This\s+will\s+help\s+substantiate\s+the\s+hypothesis\s+regarding\s+decreased\s+"
            r"trap\s+density\s+and\s+enhanced\s+charge\s+carrier\s+mobility",
            "This will characterize recombination dynamics but cannot alone assign trap density or mobility",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"This\s+prediction\s+stems\s+from\s+the\s+hypothesis\s+that[^.]*"
            r"decreases\s+trap\s+density\s+and\s+enhances\s+charge\s+carrier\s+mobility",
            "The direction of the TRPL response is unverified and cannot be uniquely assigned to "
            "trap density or mobility",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "TRPL measures photoluminescence decay and recombination dynamics; it does not alone "
            "quantify trap-state density or charge-carrier mobility."
        )

    if re.search(r"\bDSC\b|differential scanning calorimetry", guarded, re.IGNORECASE):
        guarded = re.sub(
            r"Measures\s+heat\s+changes\s+during\s+phase\s+transitions;\s*useful\s+for\s+"
            r"assessing\s+binding\s+energies",
            "Measures thermal transitions and enthalpic events but does not directly assign a "
            "spacer-lattice binding energy",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "DSC can resolve thermal transitions and enthalpic events but does not directly assign a "
            "spacer-lattice binding energy; that claim requires a defined thermodynamic cycle or "
            "validated atomistic calculation."
        )

    if re.search(r"\bITC\b|isothermal titration calorimetry", guarded, re.IGNORECASE):
        guarded = re.sub(
            r"Quantifies\s+heat\s+changes\s+during\s+molecular\s+interactions;\s*applicable\s+for\s+"
            r"studying\s+binding\s+processes",
            "Quantifies heats for defined solution-phase titrations and is not a direct measurement "
            "of binding inside a formed solid perovskite lattice",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "ITC requires a chemically defined solution-phase titration model and cannot directly "
            "measure binding within an already formed solid perovskite lattice."
        )

    if re.search(r"\bTGA\b|\bDTA\b", guarded, re.IGNORECASE):
        boundaries.append(
            "TGA and DTA can compare mass-loss and thermal-event profiles; they do not independently "
            "establish interfacial binding strength or an ion-migration mechanism."
        )

    xrd_peak_claim = bool(
        re.search(r"\bXRD\b|X-ray diffraction", guarded, re.IGNORECASE)
        and re.search(r"FWHM|full width at half maximum|narrower peaks|sharper.*peaks", guarded, re.IGNORECASE)
    )
    if xrd_peak_claim:
        guarded = re.sub(
            r"Narrower\s+peaks\s+suggest\s+higher\s+crystallinity",
            "Peak-width changes require deconvolution of coherent-domain size, microstrain, texture, and instrument broadening",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"with\s+narrower\s+peaks\s+indicating\s+higher\s+crystallinity",
            "with peak widths interpreted only after domain-size, microstrain, texture, and instrument-broadening analysis",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"where\s+a\s+higher\s+degree\s+of\s+crystallinity\s+would\s+manifest\s+as\s+"
            r"sharper\s+and\s+more\s+intense\s+peaks",
            "where peak shape, integrated intensity, texture, phase fraction, and instrumental broadening "
            "must be modeled separately",
            guarded,
            flags=re.IGNORECASE,
        )
        boundaries.append(
            "XRD peak width is affected by coherent-domain size, microstrain, texture, phase overlap, "
            "and instrument broadening; FWHM alone does not establish a directional change in crystallinity."
        )

    guarded = re.sub(
        r"X-ray\s+diffraction\s*\(XRD\)\s+analysis\s+can\s+evaluate\s+crystallinity\s+and\s+grain\s+size",
        "X-ray diffraction (XRD) can evaluate phase, texture, and peak shape; coherent-domain size "
        "requires instrument- and strain-corrected analysis and is not equivalent to film grain size",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"Examine\s+the\s+crystallinity\s+and\s+grain\s+size\s+of\s+perovskite\s+films",
        "Examine phase, texture, and peak shape, and use microscopy separately for film grain size",
        guarded,
        flags=re.IGNORECASE,
    )

    guarded = re.sub(
        r"provid(?:e|es|ing)\s+visual\s+confirmation\s+of\s+the\s+crystallinity\s+and\s+defect\s+levels",
        "providing surface-morphology and apparent-grain-size information without directly measuring "
        "crystallinity or electronic defect levels",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"Assess\s+grain\s+size,\s*crystallinity,\s*and\s*defect\s+density",
        "Assess surface morphology and apparent grain size, with diffraction and independent "
        "defect-sensitive measurements used for the other claims",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"The\s+rationale\s+behind\s+these\s+specific\s+tests\s+is\s+that\s+they\s+directly\s+"
        r"measure\s+the\s+parameters\s+predicted\s+by\s+the\s+hypothesis",
        "These measurements address complementary proxies rather than directly proving the proposed mechanism",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"The\s+proposed\s+tests\s+using\s+([^.]*)\s+are\s+appropriate\s+but",
        r"The proposed measurements using \1 are candidate tests, but",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"The\s+proposed\s+tests\s+are\s+well[- ]designed\s+and\s+appropriate\s+for\s+"
        r"evaluating\s+the\s+predictions",
        "The proposed measurements address parts of the predictions but require the stated controls "
        "and complementary methods",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"The\s+proposed\s+tests\s+are\s+well[- ]designed\s+to\s+provide\s+"
        r"specific,?\s+measurable\s+data",
        "The proposed measurements address distinct observables but require the stated controls "
        "and complementary methods",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"These\s+methods\s+align\s+with\s+the\s+properties\s+being\s+investigated\s+and\s+can\s+"
        r"provide\s+quantitative\s+data\s+to\s+support\s+or\s+refute\s+the\s+hypothesis",
        "These methods address distinct observables; none alone establishes the proposed mechanism",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"The\s+proposed\s+tests\s+are\s+appropriate\s+and\s+can\s+provide\s+the\s+necessary\s+"
        r"evidence\s+to\s+evaluate\s+the\s+hypothesis",
        "The proposed measurements are candidate tests that require the stated controls and "
        "complementary evidence to evaluate the hypothesis",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"These\s+methods\s+are\s+appropriate\s+for\s+assessing\s+([^.]*)",
        r"These methods address \1 but require method-specific controls and complementary measurements",
        guarded,
        flags=re.IGNORECASE,
    )

    guarded = re.sub(
        r"a\s+series\s+of\s+well[- ]defined\s+experiments",
        "a candidate set of experiments requiring method-specific controls",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"Techniques\s+such\s+as\s+([^.]*)\s+are\s+suitable\s+for\s+these\s+evaluations",
        r"The listed techniques (\1) address different observables and are not interchangeable "
        "or individually sufficient for these evaluations",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"proposed\s+experimental\s+tests\s+are\s+appropriate\s+for\s+validating\s+the\s+hypothesis",
        "proposed experiments are candidate tests whose interpretations require the stated controls",
        guarded,
        flags=re.IGNORECASE,
    )

    guarded = re.sub(
        r"(?:current[- ]voltage\s*\(I-V\)|I-V)\s+(?:curve\s+)?measurements?",
        "current-density-voltage (J-V) measurements with stabilized maximum-power-point output and EQE consistency checks",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"(?:The\s+)?(?:hypothesis|proposal|experimental\s+design)\s+is\s+highly\s+testable",
        "The hypothesis is conditionally testable once the controls and measurement definitions below are specified",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"\bIt\s+is\s+highly\s+testable\b",
        "It is conditionally testable once the controls and measurement definitions are specified",
        guarded,
        flags=re.IGNORECASE,
    )
    if re.search(r"\bTOF\b|time-of-flight", guarded, re.IGNORECASE) and "tof-sims" not in guarded.lower():
        boundaries.append(
            "Carrier time-of-flight must be distinguished explicitly from ToF-SIMS and requires a "
            "defined sample thickness, electric field, contact geometry, transit-time model, and "
            "checks that the transit is resolvable in the film geometry."
        )
    guarded = re.sub(
        r"(?:Each|All)\s+of\s+(?:these|the)\s+(?:tests|methods)\s+(?:is|are)\s+"
        r"well[- ]defined\s+and\s+directly\s+measures?[^.]*",
        "These tests address relevant observables, but no single measurement directly proves the proposed mechanism",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"proposed experimental tests are well-defined and capable of validating or refuting the hypothesis",
        "proposed experiments are potentially informative but require the stated controls and complementary measurements to test the hypothesis",
        guarded,
        flags=re.IGNORECASE,
    )

    pea_claim = bool(
        re.search(r"\bphenethylammonium\b|\bPEA(?:\+|\u207a)?\b", guarded, re.IGNORECASE)
    )
    if pea_claim:
        guarded = re.sub(
            r"PEA(?:'s|\u2019s)\s+flexible\s+alkyl\s+chain",
            "PEA's phenethyl framework",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:forming?|facilitates?|through)\s+strong\s+hydrogen[- ]bond(?:ing|s)?",
            "through a proposed hydrogen-bonding geometry",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"\bstrong\s+hydrogen[- ]bond(?:ing|s)?\s+interactions?",
            "proposed hydrogen-bonding interactions",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:enable|form|produce)s?\s+strong\s+interactions?\s+with\s+the\s+perovskite\s+lattice",
            "enable a proposed interfacial interaction geometry with the perovskite lattice",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"(?:Similarly,?\s*)?(?:the\s+)?conjugated\s+(?:\u03c0|pi)[- ]system\s+of\s+PEA\s+"
            r"(?:may\s+)?(?:facilitates?|enables?)\s+"
            r"charge\s+transport\s+through\s+hopping\s+or\s+tunneling\s+processes?\.?",
            "The transport barrier associated with the PEA interlayer remains unverified and should "
            "be measured using thickness-dependent transport and independent energy-alignment controls.",
            guarded,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"formation\s+of\s+a\s+dense,?\s+ordered\s+organic[- ]inorganic\s+hybrid\s+structure\s+"
            r"facilitated\s+by\s+the\s+bulky\s+PEA\s+cation",
            "formation of an interfacial phase whose composition, thickness, texture, and packing must be measured",
            guarded,
            flags=re.IGNORECASE,
        )
        if re.search(r"hydrogen[- ]bond|pi[-\s]*pi|\u03c0[-\s]*\u03c0", guarded, re.IGNORECASE):
            boundaries.append(
                "PEA hydrogen-bonding and aromatic-packing geometries are mechanistic hypotheses, "
                "not established interaction strengths; test them with structure-sensitive measurements "
                "and matched molecular controls."
            )
        if re.search(
            r"controls?\s+(?:samples?\s+)?without\s+(?:PEA|the\s+(?:PEA\s+)?interlayer)",
            guarded,
            re.IGNORECASE,
        ):
            boundaries.append(
                "A PEA-interlayer versus no-interlayer comparison changes composition and processing, "
                "so it cannot isolate the PEA mechanism; include sham-processed, thickness-matched, "
                "and alternative-spacer controls with the interfacial phase verified."
            )

    generic_sims_rate_claim = bool(
        re.search(r"\b(?:SIMS|secondary\s+ion\s+mass\s+spectrometry)\b", guarded, re.IGNORECASE)
        and re.search(r"ion\s+migration\s+rates?|direct\s+evidence\s+of\s+ion\s+migration", guarded, re.IGNORECASE)
    )
    if generic_sims_rate_claim:
        guarded = re.sub(
            r"(?i)Utilize\s+time-resolved\s+X-ray\s+diffraction\s*\(TR-XRD\)\s+and\s+"
            r"secondary\s+ion\s+mass\s+spectrometry\s*\(SIMS\)\s+to\s+monitor\s+the\s+"
            r"movement\s+of\s+ions\s+across\s+the\s+interface",
            "Use time-resolved X-ray diffraction (TR-XRD) to monitor stress-dependent structural "
            "changes and calibrated secondary ion mass spectrometry (SIMS) depth profiles to map "
            "ion redistribution across the interface",
            guarded,
        )
        guarded = re.sub(
            r"(?i)These\s+techniques\s+will\s+provide\s+direct\s+evidence\s+of\s+ion\s+"
            r"migration\s+rates[^.]*\.",
            "These measurements do not independently yield a migration rate; kinetics require "
            "controlled time, bias, and temperature series, depth-profile calibration, and a "
            "complementary transient-current or impedance measurement.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)The\s+key\s+parameter\s+(?:here\s+)?is\s+the\s+rate\s+of\s+ion\s+"
            r"displacement\s+over\s+time,?\s+which\s+will\s+provide\s+direct\s+evidence[^.]*\.",
            "Stress-dependent ion redistribution will be compared at predefined times; extracting "
            "a migration rate requires a validated kinetic model and complementary transport data.",
            guarded,
        )
        boundaries.append(
            "SIMS maps composition or stress-induced ion redistribution, while TR-XRD tracks "
            "structural evolution; neither alone directly measures an ion-migration rate."
        )

    xrd_sem_proxy_claim = bool(
        re.search(r"\bXRD\b|X-ray\s+diffraction", guarded, re.IGNORECASE)
        and re.search(r"\bSEM\b|scanning\s+electron\s+microscopy", guarded, re.IGNORECASE)
        and re.search(r"improved\s+crystallinity|fewer\s+defects|defect\s+levels?", guarded, re.IGNORECASE)
    )
    if xrd_sem_proxy_claim:
        guarded = re.sub(
            r"(?i)(?:The\s+)?[^.]{0,120}(?:treated|interfaced)\s+samples?\s+(?:are\s+)?"
            r"expected\s+to\s+show\s+improved\s+crystallinity\s+and\s+fewer\s+defects[^.]*\.",
            "The samples will be tested for differences in phase, texture, peak shape, and surface "
            "morphology; XRD and SEM alone cannot assign electronic defect-state density.",
            guarded,
        )
        boundaries.append(
            "XRD and SEM can assess phase, texture, peak shape, and morphology, but they do not "
            "alone establish improved crystallinity or quantify electronic defect-state density."
        )

    directional_pce_claim = bool(
        re.search(
            r"(?i)(?:increase|improvement)\s+in\s+(?:the\s+)?(?:PCE|power\s+conversion\s+efficiency)|"
            r"(?:PCE|power\s+conversion\s+efficiency)[^.]{0,120}(?:predicted|expected)\s+to\s+"
            r"(?:increase|be\s+(?:measurably\s+)?higher)|"
            r"assess\s+any\s+improvements?\s+in\s+efficiency",
            guarded,
        )
        and re.search(r"ion\s+migration|no\s+direct|remains\s+unverified", guarded, re.IGNORECASE)
    )
    if directional_pce_claim:
        guarded = re.sub(
            r"(?i)(?:an?\s+)?increase\s+in\s+(?:the\s+)?power\s+conversion\s+efficiency\s*"
            r"\(PCE\)[^.]*\.",
            "PCE will be reported as a secondary device-level outcome without a prespecified "
            "direction and cannot validate the proposed ion-transport mechanism.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)(?:the\s+)?(?:PCE|power\s+conversion\s+efficiency)[^.]{0,120}?\s+is\s+"
            r"(?:predicted|expected)\s+to\s+(?:increase|be\s+(?:measurably\s+)?higher)[^.]*\.",
            "PCE will be reported as a secondary device-level outcome without a prespecified "
            "direction and cannot validate the proposed ion-transport mechanism.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)assess\s+any\s+improvements?\s+in\s+efficiency",
            "assess any difference in device performance without assigning its direction",
            guarded,
        )
        boundaries.append(
            "PCE is a secondary system-level outcome and must not be assigned a direction or used "
            "to validate the proposed ion-migration mechanism without direct support."
        )

    tpv_tpc_proxy_claim = bool(
        re.search(r"\bTPV\b|transient\s+photovoltage", guarded, re.IGNORECASE)
        and re.search(r"mobilit|carrier\s+lifetime|charge\s+transport\s+efficiency", guarded, re.IGNORECASE)
    )
    if tpv_tpc_proxy_claim:
        guarded = re.sub(
            r"(?i)(?:the\s+)?decay\s+times\s+and\s+magnitudes\s+of\s+the\s+transient\s+"
            r"signals,?\s+which\s+are\s+indicative\s+of\s+charge\s+carrier\s+lifetimes\s+and\s+"
            r"mobilities",
            "the device-level transient responses, which reflect coupled recombination, extraction, "
            "capacitance, and contact effects rather than intrinsic lifetime or mobility alone",
            guarded,
        )
        guarded = re.sub(
            r"(?i)(?:Transient\s+Photovoltage\s+Measurements|TPV)\s*:\s*These\s+measurements\s+"
            r"will\s+assess\s+the\s+charge\s+transport\s+efficiency",
            "Transient photovoltage measurements: These device-level transients will test for a "
            "response difference but will not independently assign transport efficiency or mobility",
            guarded,
        )
        guarded = re.sub(
            r"(?i)By\s+comparing\s+the\s+decay\s+times\s+of\s+photogenerated\s+carriers[^.]{0,180}\.",
            "Decay-time differences will be interpreted only with controls for recombination, "
            "extraction, capacitance, contacts, and operating point.",
            guarded,
        )
        guarded = re.sub(
            r"(?i)help\s+determine\s+if\s+the\s+PEA\s+interlayer\s+affects\s+charge\s+carrier\s+mobility",
            "test whether the device-level transient response changes without assigning intrinsic mobility",
            guarded,
        )
        boundaries.append(
            "TPV and TPC are device-level transient responses affected by recombination, extraction, "
            "capacitance, contacts, and operating point; they do not independently measure intrinsic "
            "carrier lifetime or mobility."
        )

    guarded = re.sub(
        r"(?i)current[- ]voltage\s*\(I-V\)\s+curve\s+analysis",
        "current-density-voltage (J-V) sweeps with stabilized maximum-power-point output and EQE consistency checks",
        guarded,
    )
    guarded = re.sub(r"\bFAPB3\b", "FAPbI3", guarded)
    guarded = re.sub(
        r"(?i)(?:the\s+prediction\s+that\s+)?(?:the\s+)?interlayer\s+will\s+maintain\s+its\s+"
        r"structural\s+integrity\s+and\s+functionality\s+up\s+to\s+80\s*(?:\u00b0\s*)?C\s+is\s+testable\.",
        "An 80 C exposure can be used as a prespecified stress condition, but structural integrity "
        "requires before/after structural and chemical measurements and functionality requires an "
        "independent operational device test.",
        guarded,
    )
    guarded = re.sub(
        r"(?i)(?:this\s+(?:specific\s+)?application|the\s+hypothesis)\s+appears\s+novel,?\s+"
        r"as\s+direct\s+supporting\s+evidence[^.]*\.",
        "Novelty cannot be inferred from an empty retrieval result and requires a broader prior-art search.",
        guarded,
    )

    guarded = re.sub(
        r"(?i)(?:the\s+)?(?:experimental\s+)?tests?\s+(?:outlined\s+)?are\s+robust\s+and\s+"
        r"well[- ]suited\s+to\s+validate\s+or\s+falsify\s+the\s+hypothesis",
        "The proposed tests are candidate measurements whose interpretations depend on the "
        "method-specific controls above",
        guarded,
    )
    guarded = re.sub(
        r"(?i)(?:the\s+)?proposed\s+tests?\s+are\s+well[- ]designed[^.]*\.",
        "The proposed tests are candidate measurements whose interpretations depend on the "
        "method-specific controls and complementary measurements above.",
        guarded,
    )
    guarded = re.sub(
        r"(?i)(?:the\s+)?proposed\s+tests?\s+are\s+well[- ]suited[^.]*\.",
        "The proposed tests are candidate measurements whose interpretations depend on the "
        "method-specific controls and complementary measurements above.",
        guarded,
    )
    guarded = re.sub(
        r"(?i)represent(?:s)?\s+a\s+novel\s+approach",
        "remain a candidate distinction whose novelty requires a broader prior-art search",
        guarded,
    )
    guarded = re.sub(
        r"(?i)retains?\s+a\s+degree\s+of\s+novelty\s+as\s+an\s+unverified\s+scientific\s+exploration",
        "has undetermined novelty pending a broader prior-art search",
        guarded,
    )
    guarded = re.sub(
        r"(?i)introduces?\s+a\s+novel\s+(?:combination|approach)",
        "defines a specific testable combination",
        guarded,
    )
    guarded = re.sub(
        r"(?i)introduces?\s+a\s+fresh\s+perspective",
        "frames a specific testable comparison",
        guarded,
    )
    guarded = re.sub(
        r"(?i)with\s+promising\s+potential",
        "requiring empirical evaluation",
        guarded,
    )
    return guarded, list(dict.fromkeys(boundaries))


def guard_hypothesis_synthesis(text: str) -> str:
    cleaned = _strip_internal_process_notes(str(clean_model_markup(text)))
    no_direct_support = _states_no_direct_support(cleaned)
    if no_direct_support:
        cleaned = _strip_citations_without_direct_support(cleaned)
    cleaned, composition_note = _composition_gate(cleaned)
    cleaned, quantitative_changed = strip_unsupported_quantitative_targets(cleaned)
    cleaned, transport_boundaries = _guard_transport_and_outcome_claims(cleaned)
    cleaned, spacer_method_boundaries = _guard_spacer_and_method_claims(cleaned)
    dj_candidate_class_note = _dj_candidate_class_boundary(cleaned)
    sections: list[str] = []
    if no_direct_support:
        sections.append(
            "**Evidence boundary**\nNo directly relevant evidence was retrieved. All compound-specific "
            "mechanisms and directional predictions below remain unverified proposals."
        )
    if composition_note:
        sections.append(f"**Chemical composition gate**\n{composition_note}")
    identity_note = _three_apb_boundary(cleaned)
    if identity_note:
        sections.append(f"**Chemical identity boundary**\n{identity_note}")
    cha_identity_note = _cha_identity_boundary(cleaned)
    if cha_identity_note:
        cleaned = _remove_unsupported_cha_assignments(cleaned)
        sections.append(f"**CHA identity boundary**\n{cha_identity_note}")
    if dj_candidate_class_note:
        sections.append(f"**DJ candidate-class gate**\n{dj_candidate_class_note}")
    if quantitative_changed:
        sections.append(
            "**Quantitative-target gate**\nUnsupported relative percentage targets were converted "
            "to directional, measurable predictions. A numerical threshold requires prior data, "
            "measurement precision, or a statistical power calculation."
        )
    if transport_boundaries:
        sections.append(
            "**Mechanism and measurement gate**\n"
            + "\n".join(f"- {item}" for item in transport_boundaries)
        )
    if spacer_method_boundaries:
        sections.append(
            "**Identity, comparison, and measurement boundaries**\n"
            + "\n".join(f"- {item}" for item in spacer_method_boundaries)
        )
    if not sections:
        return cleaned
    return "\n\n".join([*sections, cleaned]).strip()


def _method_boundaries(text: str) -> list[str]:
    boundaries: list[str] = []
    lower = text.lower()
    if "tof-sims" in lower or "tofsims" in lower:
        if "migration" in lower or "ion movement" in lower:
            boundaries.append(
                "ToF-SIMS maps ion distributions or stress-induced redistribution. A migration rate "
                "requires a controlled time/bias/temperature series and a complementary transient or "
                "impedance measurement."
            )
    if "hall effect" in lower and ("mobility" in lower or "charge carrier" in lower):
        boundaries.append(
            "Hall mobility in lead-halide perovskites requires controls for high impedance and mixed "
            "ionic-electronic conduction; photo-Hall or a complementary contactless transport method "
            "is needed for a reliable assignment."
        )
    if ("uv-vis" in lower or "uv vis" in lower) and any(
        term in lower for term in ("band alignment", "electronic structure", "interface", "bandgap shift")
    ):
        boundaries.append(
            "UV-Vis can estimate an optical absorption edge but cannot establish buried-interface "
            "band alignment or complete electronic structure; interface-sensitive photoelectron or "
            "potential measurements are required."
        )
    if "stable interface" in lower or "interface stability" in lower:
        boundaries.append(
            "Interface formation and stability must be tested independently with structural, chemical, "
            "cross-sectional, and controlled-aging measurements rather than assumed as a premise."
        )
    return boundaries


def guard_hypothesis_analysis(report: str, hypothesis_context: str = "") -> str:
    combined_original = f"{hypothesis_context}\n{report}"
    invalid_composition = bool(_CHA_INVALID_FORMULA.search(combined_original))
    had_quantitative_target = bool(re.search(_PERCENT, combined_original))
    guarded = _strip_internal_process_notes(str(clean_model_markup(report)))
    guarded = re.sub(
        r"\A\*\*Deterministic scientific audit\*\*.*?"
        r"Any broader statement below that calls the design fully appropriate, robust, or well-aligned "
        r"is superseded by this audit\.\s*",
        "",
        guarded,
        flags=re.DOTALL,
    )
    guarded = re.sub(
        r"(?im)^\s*\*\*Calibrated status:\*\*[^\n]*\s*$",
        "",
        guarded,
    ).strip()
    no_direct_support = _states_no_direct_support(combined_original)
    if no_direct_support:
        guarded = _strip_citations_without_direct_support(guarded)
    guarded, _ = _composition_gate(guarded)
    guarded, quantitative_changed = strip_unsupported_quantitative_targets(guarded)
    guarded, _ = _guard_transport_and_outcome_claims(guarded)
    guarded, spacer_method_boundaries = _guard_spacer_and_method_claims(guarded)
    _, context_transport_boundaries = _guard_transport_and_outcome_claims(combined_original)
    guarded = re.sub(
        r"tests? (?:that are|are) scientifically sound and feasible",
        "tests that are partially relevant but incomplete",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"proposed (?:tests|methods) are well[- ]aligned",
        "proposed tests are only partially aligned",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"testability (?:of the hypothesis )?is strong",
        "testability is conditional on resolving the stated definition and measurement gaps",
        guarded,
        flags=re.IGNORECASE,
    )

    boundaries: list[str] = []
    if no_direct_support:
        boundaries.append(
            "No directly relevant evidence was retrieved; background or knowledge-graph claims cannot "
            "validate the selected compound, mechanism, or directional prediction."
        )
    if invalid_composition:
        boundaries.append(
            f"The original CHA+(PbI4)2 composition was not charge-balanced. The intended n=1 RP "
            f"phase must be confirmed as {_CHA_RP_FORMULA} before interface mechanisms are evaluated."
        )
    identity_note = _three_apb_boundary(combined_original)
    if identity_note:
        boundaries.append(identity_note)
    cha_identity_note = _cha_identity_boundary(combined_original)
    if cha_identity_note:
        guarded = _remove_unsupported_cha_assignments(guarded)
        boundaries.append(cha_identity_note)
    dj_candidate_class_note = _dj_candidate_class_boundary(combined_original)
    if dj_candidate_class_note:
        boundaries.append(dj_candidate_class_note)
    boundaries.extend(_method_boundaries(combined_original))
    boundaries.extend(item for item in spacer_method_boundaries if item not in boundaries)
    boundaries.extend(
        item for item in context_transport_boundaries if item not in boundaries
    )
    if quantitative_changed or had_quantitative_target:
        boundaries.append(
            "Relative percentage targets without an empirical baseline or power calculation are not "
            "acceptance criteria and have been reduced to directional predictions."
        )
    if not boundaries:
        return guarded

    status = (
        "Definition-required; not ready for mechanistic evaluation."
        if invalid_composition or identity_note or cha_identity_note or dj_candidate_class_note
        else "Speculative but testable only after the listed measurement gaps are resolved."
    )
    audit = "\n".join(f"- {item}" for item in boundaries)
    return (
        f"**Deterministic scientific audit**\n{audit}\n\n"
        "Any broader statement below that calls the design fully appropriate, robust, or well-aligned "
        "is superseded by this audit.\n\n"
        f"{guarded}\n\n**Calibrated status:** {status}"
    ).strip()


def wrap_clean_model_output(function: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return clean_model_markup(await function(*args, **kwargs))

        return async_wrapper

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return clean_model_markup(function(*args, **kwargs))

    return wrapper
