"""Pure, read-only validation for experimental planning drafts."""

from __future__ import annotations

from typing import Any


def experiment_automation_blockers(constraints: dict[str, Any], plan: str) -> list[str]:
    """Return release blockers without generating artifacts or contacting equipment."""
    liquid = constraints.get("liquid_handling") or {}
    materials = [str(item).strip() for item in liquid.get("materials") or [] if str(item).strip()]
    instruments = [str(item).strip() for item in liquid.get("instruments") or [] if str(item).strip()]
    equipment = [str(item).strip() for item in constraints.get("equipment") or [] if str(item).strip()]
    techniques = [str(item).strip() for item in constraints.get("techniques") or [] if str(item).strip()]
    blockers: list[str] = []

    if not materials:
        blockers.append("Configure the exact material and stock-solution identities.")
    if not instruments:
        blockers.append("Configure a liquid-handling method, including manual pipettes when applicable.")
    if not techniques:
        blockers.append("Configure the characterization techniques used for support and falsification.")

    plan_lower = str(plan or "").lower()
    configured = " ".join([*instruments, *equipment, *techniques]).lower()
    for marker, label in (
        ("tecan", "Tecan liquid handler"),
        ("opentrons", "Opentrons liquid handler"),
        ("spin coat", "spin coating equipment"),
        ("xrd", "XRD"),
        ("giwaxs", "GIWAXS"),
        ("sem", "SEM"),
        ("tem", "TEM"),
    ):
        if marker in plan_lower and marker not in configured:
            blockers.append(f"The plan names unconfigured equipment or characterization: {label}.")

    if "control" not in plan_lower:
        blockers.append("Define negative, process-matched, and mechanism-discriminating controls.")
    if not any(term in plan_lower for term in ("replicate", "triplicate", "n=")):
        blockers.append("Define biological or technical replicates for each condition.")
    if "random" not in plan_lower:
        blockers.append("Define well/sample randomization or a justified blocking scheme.")

    interfacial = any(
        term in plan_lower for term in ("interlayer", "interface", "thin-film", "thin film", "device")
    )
    if interfacial:
        if not any(
            term in plan_lower
            for term in ("substrate", "film", "device", "deposit", "spin-coat", "spin coat")
        ):
            blockers.append(
                "Specify how the physical film or device interface is fabricated; solution mixing is not an interface."
            )
        if not any(
            term in plan_lower for term in ("phase verification", "xrd", "giwaxs", "grazing-incidence")
        ):
            blockers.append("Include a configured method to verify the claimed interfacial phase.")
        if "sham" not in plan_lower:
            blockers.append("Include a sham-processed control.")
        if "thickness-matched" not in plan_lower:
            blockers.append("Include a thickness-matched control.")
        if "alternative spacer" not in plan_lower:
            blockers.append("Include an alternative-spacer control.")

    return list(dict.fromkeys(blockers))
