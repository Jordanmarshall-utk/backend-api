from __future__ import annotations

import inspect
import re
import unicodedata
from functools import wraps
from typing import Any, Callable

from app.services.literature_kg_retrieval import build_hypothesis_kg_context


_GROUNDING_MARKER = "LITERATURE_KG_CONTEXT"
_AUDIT_MARKER = "Branch audit:"
_IDENTITY_RETRY_MARKER = "IDENTITY_RETRY_BOUNDARY"
_CATION_RE = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9-]{1,14})\s*(?:\+|\u207a)")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_KNOWN_CATIONS = {
    "BA",
    "CA",
    "CS",
    "EA",
    "FA",
    "K",
    "MA",
    "NA",
    "PA",
    "PEA",
    "RB",
}
_TEXT_ARGUMENTS = (
    "question",
    "user_question",
    "initial_clarified_question",
    "clarified_question",
    "socratic_question",
    "prompt",
    "hypothesis",
    "topic",
    "query",
    "message",
)
_BRANCH_KEYS = {"branches", "choices", "hypotheses", "options", "response"}


def normalize_choice_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u00a0": " ",
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
                "\u207a": "+",
            }
        )
    )
    return " ".join(normalized.split()).casefold()


def _ambiguous_identifiers(source: str) -> list[str]:
    return sorted(
        {
            token
            for token in _CATION_RE.findall(source)
            if token.upper() not in _KNOWN_CATIONS
        }
    )


def _findings(source: str, result_text: str) -> list[str]:
    findings: list[str] = []
    ambiguous = _ambiguous_identifiers(source)
    if ambiguous:
        findings.append(
            "Treat "
            + ", ".join(ambiguous)
            + " as proposed identifiers until name, formula, structure, charge, and "
            "protonation state are supplied; novelty is allowed, invented identity is not."
        )
    source_percentages = set(_PERCENT_RE.findall(source))
    introduced_percentages = set(_PERCENT_RE.findall(result_text)) - source_percentages
    if introduced_percentages:
        findings.append(
            "New quantitative targets "
            + ", ".join(sorted(introduced_percentages))
            + " are exploratory predictions rather than literature-supported thresholds."
        )
    lowered = result_text.casefold()
    method_boundaries: list[str] = []
    if "tof-sims" in lowered and "migration rate" in lowered:
        method_boundaries.append("ToF-SIMS needs a time-resolved design to infer migration rates")
    if "xrd" in lowered and ("bond strength" in lowered or "stronger bond" in lowered):
        method_boundaries.append("XRD spacing alone does not establish bond strength")
    if "afm" in lowered and "defect density" in lowered:
        method_boundaries.append("AFM morphology is not a direct defect-density measurement")
    if "steady-state pl" in lowered and "lifetime" in lowered:
        method_boundaries.append("steady-state PL does not measure carrier lifetime")
    if method_boundaries:
        findings.append("; ".join(method_boundaries) + ".")
    return findings


def _identifier_pattern(identifier: str) -> str:
    return (
        rf"(?<![A-Za-z0-9-]){re.escape(identifier)}\s*(?:\+|\u207a)?"
        r"(?![A-Za-z0-9-])"
    )


def _branch_has_identity_violation(text: str, source: str) -> bool:
    ambiguous = _ambiguous_identifiers(source)
    if not ambiguous:
        return False
    lowered = unicodedata.normalize("NFKC", text).casefold()
    source_lowered = unicodedata.normalize("NFKC", source).casefold()
    for identifier in ambiguous:
        identifier_pattern = _identifier_pattern(identifier)
        if not re.search(identifier_pattern, text, flags=re.IGNORECASE):
            return True
        if re.search(
            rf"{identifier_pattern}\s+(?:is|was|will\s+be)\s+"
            r"(?:assigned|defined|identified|specified)\s+(?:as\s+)?(?:an?\s+)?"
            r"(?:new\s+)?(?:chemical\s+)?(?:identity|structure|formula|compound|cation)",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            rf"{identifier_pattern}(?:['\u2019]s)?\s+"
            r"(?:fluorine|halogen|hydroxyl|amino|amine|ether|carbonyl|carboxyl|"
            r"sulfonyl|alkyl\s+chain|methylene\s+chain|molecular\s+formula)",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        unsupported_descriptors = (
            "electron-rich",
            "electron rich",
            "fluorinated",
            "fluorine substituent",
            "hydroxyl group",
            "alkyl chain",
            "methylene chain",
            "dicationic",
            "bulky",
            "small size",
            "steric hindrance",
        )
        nearby_identifier = re.search(
            rf"{identifier_pattern}[^.]{{0,220}}",
            text,
            flags=re.IGNORECASE,
        )
        if nearby_identifier:
            nearby_text = unicodedata.normalize(
                "NFKC", nearby_identifier.group(0)
            ).casefold()
            if any(
                descriptor in nearby_text and descriptor not in source_lowered
                for descriptor in unsupported_descriptors
            ):
                return True
    if re.search(
        r"(?:\u03c0\s*[-\u2010-\u2015]?\s*\u03c0|pi\s*[-\u2010-\u2015]?\s*pi)\s+"
        r"stack\w*[^.]{0,140}perovskite\s+(?:lattice|framework|layers?)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"(?:aromatic\s+ring|molecular\s+orientation)[^.]{0,140}"
        r"(?:align\w*|match\w*|coupl\w*)[^.]{0,100}"
        r"(?:conduction|valence)\s+band",
        lowered,
        flags=re.IGNORECASE,
    ):
        return True
    source_cations = {token.casefold() for token in _CATION_RE.findall(source)}
    new_cations = {
        token
        for token in _CATION_RE.findall(text)
        if token.casefold() not in source_cations
        and token.upper() not in _KNOWN_CATIONS
    }
    comparison_markers = (
        "analog",
        "analogy",
        "control",
        "comparator",
        "reference compound",
    )
    return bool(new_cations) and not any(
        marker in lowered for marker in comparison_markers
    )


def _generated_branch_strings(result: Any) -> list[str]:
    if isinstance(result, tuple) and len(result) == 2:
        options = result[1]
        if isinstance(options, (list, tuple)):
            return [item for item in options if isinstance(item, str)]
    if isinstance(result, (list, tuple)):
        return [item for item in result if isinstance(item, str)]
    if isinstance(result, dict):
        branches: list[str] = []
        for key, value in result.items():
            if str(key).casefold() in _BRANCH_KEYS:
                branches.extend(_generated_branch_strings(value))
        return branches
    if isinstance(result, str):
        return [result]
    return []


def _needs_identity_retry(result: Any, source: str) -> bool:
    branches = _generated_branch_strings(result)
    return bool(branches) and any(
        _branch_has_identity_violation(branch, source) for branch in branches
    )


def _identity_retry_arguments(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    source: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    identifiers = ", ".join(f"{item}+" for item in _ambiguous_identifiers(source))
    instruction = (
        f"{_IDENTITY_RETRY_MARKER}\n"
        f"The prior branches failed the identity or mechanism boundary for {identifiers}. "
        "Regenerate exactly three mechanistically distinct branches about the supplied identifier itself. "
        "Every branch must name that identifier and keep its name, formula, detailed structure, charge state, "
        "and substituents unresolved unless the user supplied them. Do not substitute a named molecule. "
        "Do not infer electron richness, substituents, molecular size, steric hindrance, a pi-stacking "
        "partner, or band alignment from an unresolved identity. Mechanisms must be physically coherent "
        "and linked to claim-appropriate tests. Known compounds may appear only as explicitly labeled "
        "analogies, controls, or comparators."
    )
    signature = inspect.signature(function)
    bound = signature.bind_partial(*args, **kwargs)
    argument_name = next(
        (
            name
            for name in _TEXT_ARGUMENTS
            if isinstance(bound.arguments.get(name), str)
        ),
        "",
    )
    if argument_name:
        bound.arguments[argument_name] = (
            str(bound.arguments[argument_name]) + "\n\n" + instruction
        )
    return bound.args, bound.kwargs


def _identity_safe_fallback_branches(source: str) -> list[str]:
    identifiers = _ambiguous_identifiers(source)
    identifier = f"{identifiers[0]}+" if identifiers else "the proposed cation"
    boundary = (
        f"Evidence boundary: {identifier} remains chemically undefined, so the mechanism is "
        "prospective and no substituent, formula, quantitative target, or identity-specific evidence is assumed."
    )
    return [
        (
            f"Interfacial ion-association branch: Test whether {identifier} increases the effective "
            "activation barrier for mobile ionic defects through a measurable change in interfacial "
            "electrostatic association, while evaluating charge extraction independently. Use matched "
            "controls and bias-, time-, and temperature-dependent ion-transport measurements. "
            + boundary
        ),
        (
            f"Interfacial packing branch: Test whether incorporating {identifier} changes interfacial "
            "packing or the continuity of ion-transport pathways without assuming its size, shape, or "
            "substituents. Correlate structural and cross-sectional microscopy observations with dedicated "
            "ion-transport measurements rather than treating structure alone as migration proof. "
            + boundary
        ),
        (
            f"Defect-electronic tradeoff branch: Test whether {identifier} changes the population or "
            "activity of mobile charged defects while preserving interfacial energetics and carrier "
            "extraction. Measure ionic kinetics and electronic extraction with separate methods and leave "
            "both effect direction and magnitude open. "
            + boundary
        ),
    ]


def _replace_generated_options_with_fallback(result: Any, source: str) -> Any:
    fallback = _identity_safe_fallback_branches(source)
    if isinstance(result, tuple) and len(result) == 2:
        prompt_text, options = result
        replacement = tuple(fallback) if isinstance(options, tuple) else fallback
        return prompt_text, replacement
    if isinstance(result, list):
        return fallback
    if isinstance(result, tuple):
        return tuple(fallback)
    if isinstance(result, dict):
        updated = dict(result)
        for key in tuple(updated):
            if str(key).casefold() in _BRANCH_KEYS:
                updated[key] = _replace_generated_options_with_fallback(
                    updated[key], source
                )
        return updated
    if isinstance(result, str):
        return fallback[0]
    return result


def annotate_branch_payload(payload: Any, source: str) -> Any:
    if isinstance(payload, str):
        if _AUDIT_MARKER in payload or len(payload.strip()) < 30:
            return payload
        findings = _findings(source, payload)
        if not findings:
            return payload
        return payload.rstrip() + "\n\n" + _AUDIT_MARKER + " " + " ".join(findings)
    if isinstance(payload, list):
        return [annotate_branch_payload(item, source) for item in payload]
    if isinstance(payload, tuple):
        return tuple(annotate_branch_payload(item, source) for item in payload)
    if isinstance(payload, dict):
        return {
            key: annotate_branch_payload(value, source)
            if str(key).casefold() in _BRANCH_KEYS
            else value
            for key, value in payload.items()
        }
    return payload


def _ground_arguments(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any], str]:
    try:
        signature = inspect.signature(function)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return args, kwargs, ""
    argument_name = next(
        (
            name
            for name in _TEXT_ARGUMENTS
            if isinstance(bound.arguments.get(name), str)
        ),
        "",
    )
    if not argument_name:
        return args, kwargs, ""
    source = str(bound.arguments[argument_name])
    if _GROUNDING_MARKER in source:
        return args, kwargs, source
    context = build_hypothesis_kg_context(source)
    boundary = (
        "NOVELTY_BOUNDARY\n"
        "Novel compounds and mechanisms may be proposed. Keep undefined chemical "
        "identities unresolved, distinguish retrieved evidence from analogy and "
        "prediction, and do not convert directional literature claims into causal proof. "
        "When the question supplies an undefined identifier, every branch must retain that "
        "identifier and may not replace it with a named molecule or invented structure. Do not infer "
        "identity-dependent electronic, steric, substituent, packing, or band-alignment properties."
    )
    addition = "\n\n".join(item for item in (context, boundary) if item)
    bound.arguments[argument_name] = source + "\n\n" + addition
    return bound.args, bound.kwargs, source


def _ground_callable(
    function: Callable[..., Any], annotate_result: bool
) -> Callable[..., Any]:
    if getattr(function, "__polaris_kg_grounded__", False):
        return function
    if inspect.isasyncgenfunction(function):
        @wraps(function)
        async def async_generator(*args: Any, **kwargs: Any):
            grounded_args, grounded_kwargs, source = _ground_arguments(
                function, args, kwargs
            )
            async for item in function(*grounded_args, **grounded_kwargs):
                if annotate_result and _needs_identity_retry(item, source):
                    item = _replace_generated_options_with_fallback(item, source)
                yield _annotate_generated_options(item, source) if annotate_result else item

        wrapped = async_generator
    elif inspect.iscoroutinefunction(function):
        @wraps(function)
        async def coroutine(*args: Any, **kwargs: Any):
            grounded_args, grounded_kwargs, source = _ground_arguments(
                function, args, kwargs
            )
            result = await function(*grounded_args, **grounded_kwargs)
            if annotate_result and _needs_identity_retry(result, source):
                retry_args, retry_kwargs = _identity_retry_arguments(
                    function, grounded_args, grounded_kwargs, source
                )
                result = await function(*retry_args, **retry_kwargs)
                if _needs_identity_retry(result, source):
                    result = _replace_generated_options_with_fallback(result, source)
            return _annotate_generated_options(result, source) if annotate_result else result

        wrapped = coroutine
    else:
        @wraps(function)
        def synchronous(*args: Any, **kwargs: Any):
            grounded_args, grounded_kwargs, source = _ground_arguments(
                function, args, kwargs
            )
            result = function(*grounded_args, **grounded_kwargs)
            if annotate_result and _needs_identity_retry(result, source):
                retry_args, retry_kwargs = _identity_retry_arguments(
                    function, grounded_args, grounded_kwargs, source
                )
                result = function(*retry_args, **retry_kwargs)
                if _needs_identity_retry(result, source):
                    result = _replace_generated_options_with_fallback(result, source)
            return _annotate_generated_options(result, source) if annotate_result else result

        wrapped = synchronous
    setattr(wrapped, "__polaris_kg_grounded__", True)
    return wrapped


def _annotate_generated_options(result: Any, source: str) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        prompt_text, options = result
        if isinstance(options, (list, tuple)):
            return prompt_text, annotate_branch_payload(options, source)
    return annotate_branch_payload(result, source)


def install_hypothesis_grounding(namespace: dict[str, Any]) -> None:
    for name in (
        "clarify_question",
        "socratic_pass_and_tot",
        "retry_thinking_deepen_thoughts",
        "hypothesis_synthesis",
    ):
        function = namespace.get(name)
        if callable(function):
            namespace[name] = _ground_callable(
                function,
                annotate_result=name
                in {"socratic_pass_and_tot", "retry_thinking_deepen_thoughts"},
            )


def _candidate_strings(value: Any, depth: int = 0):
    if isinstance(value, str):
        yield value
    elif depth < 2 and isinstance(value, dict):
        for item in value.values():
            yield from _candidate_strings(item, depth + 1)
    elif depth < 2 and isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _candidate_strings(item, depth + 1)


def _unicode_safe_resolver(function: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(function, "__polaris_unicode_safe__", False):
        return function
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        choice_name = next(
            (
                name
                for name in ("choice", "selection", "selected", "answer")
                if isinstance(bound.arguments.get(name), str)
            ),
            "",
        )
        if not choice_name:
            return function(*args, **kwargs)
        choice = str(bound.arguments[choice_name])
        normalized = normalize_choice_text(choice)
        candidates: list[str] = []
        for name, value in bound.arguments.items():
            if name != choice_name:
                candidates.extend(_candidate_strings(value))
        owner = next(iter(bound.arguments.values()), None)
        for attribute in ("branches", "choices", "options", "hypotheses"):
            if owner is not None and hasattr(owner, attribute):
                candidates.extend(_candidate_strings(getattr(owner, attribute)))
        match = next(
            (item for item in candidates if normalize_choice_text(item) == normalized),
            None,
        )
        if match is not None:
            bound.arguments[choice_name] = match
        elif normalized in {"1", "2", "3"}:
            bound.arguments[choice_name] = normalized
        return function(*bound.args, **bound.kwargs)

    setattr(wrapped, "__polaris_unicode_safe__", True)
    return wrapped


def install_unicode_choice_resolution(namespace: dict[str, Any]) -> None:
    module_name = namespace.get("__name__")
    module_function = namespace.get("_resolve_choice")
    if callable(module_function):
        namespace["_resolve_choice"] = _unicode_safe_resolver(module_function)
    for value in tuple(namespace.values()):
        if not isinstance(value, type) or value.__module__ != module_name:
            continue
        descriptor = value.__dict__.get("_resolve_choice")
        if isinstance(descriptor, staticmethod):
            setattr(
                value,
                "_resolve_choice",
                staticmethod(_unicode_safe_resolver(descriptor.__func__)),
            )
        elif isinstance(descriptor, classmethod):
            setattr(
                value,
                "_resolve_choice",
                classmethod(_unicode_safe_resolver(descriptor.__func__)),
            )
        elif callable(descriptor):
            setattr(value, "_resolve_choice", _unicode_safe_resolver(descriptor))
