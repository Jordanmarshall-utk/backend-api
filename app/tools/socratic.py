import os
import sys
import logging
import re
from contextvars import ContextVar

# Fix Windows console encoding for Unicode characters (e.g., superscripts like ⁺)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Older Python or non-standard streams
    # Also configure logging to handle Unicode
    for handler in logging.root.handlers[:]:
        if hasattr(handler, 'stream') and handler.stream in (sys.stdout, sys.stderr):
            try:
                handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

# Lazy import heavy dependencies - only load when actually needed
_edison_client = None
_edison_models = None
_given_vars = None
_reasoning_layer_build_fn = None
_reasoning_layer_gen_fn = None
_last_literature_handoff: ContextVar[dict | None] = ContextVar(
    "last_literature_handoff", default=None
)


class LiteratureEvidenceUnavailableError(RuntimeError):
    """Backward-compatible error for callers that explicitly require evidence."""


def _lazy_import_reasoning_layer():
    global _reasoning_layer_build_fn
    if _reasoning_layer_build_fn is None:
        from app.tools.scientific_reasoning_policy_layer import build_reasoning_prompt
        _reasoning_layer_build_fn = build_reasoning_prompt
    return _reasoning_layer_build_fn


def _apply_reasoning_policy(base_prompt: str, task: str, task_type: str, policy: str = "auto") -> tuple:
    """Wrap base_prompt with the selected reasoning policy. Falls back gracefully on any error."""
    if not policy or policy.strip().lower() in ("none", "off", "disabled"):
        return base_prompt, {"reasoning_layer_enabled": False}
    try:
        build_fn = _lazy_import_reasoning_layer()
        return build_fn(
            base_prompt=base_prompt,
            task=task,
            agent_name="Hypothesis Agent",
            task_type=task_type,
            policy=policy,
        )
    except Exception as exc:
        logging.warning(f"Reasoning policy layer unavailable, using base prompt: {exc}")
        return base_prompt, {"reasoning_layer_enabled": False}


def _lazy_import_reasoning_gen():
    global _reasoning_layer_gen_fn
    if _reasoning_layer_gen_fn is None:
        from app.tools.scientific_reasoning_policy_layer import generate_with_reasoning
        _reasoning_layer_gen_fn = generate_with_reasoning
    return _reasoning_layer_gen_fn


def _log_reasoning_metadata(metadata: dict, fn_name: str = "") -> None:
    """Persist reasoning metadata to a sidecar log without affecting return values."""
    try:
        import time
        from app.tools.scientific_reasoning_policy_layer import dump_reasoning_metadata_json
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "reasoning")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"reasoning_{fn_name}_{int(time.time() * 1000)}.json")
        dump_reasoning_metadata_json(metadata, path)
    except Exception:
        pass


def _generate_with_reasoning_wrapper(
    base_prompt: str,
    task: str,
    task_type: str,
    policy: str = "auto",
    fn_name: str = "",
) -> dict:
    """
    Full generate_with_reasoning integration for literature-consuming LLM calls.
    Falls back to plain generate_text_with_llm if the layer is unavailable.
    Saves reasoning_metadata to a sidecar log; never modifies the str response.
    Returns a dict with at least a 'response' key.
    """
    try:
        gen_fn = _lazy_import_reasoning_gen()
        result = gen_fn(
            generate_text_fn=generate_text_with_llm,
            base_prompt=base_prompt,
            task=task,
            agent_name="Literature Agent",
            task_type=task_type,
            policy=policy,
        )
        _log_reasoning_metadata(result.get("reasoning_metadata", {}), fn_name=fn_name)
        return result
    except Exception as exc:
        logging.warning("generate_with_reasoning unavailable for %s, using base prompt: %s", fn_name, exc)
        return {"response": generate_text_with_llm(base_prompt), "reasoning_metadata": {}}

# Initialize API key from environment
QWEN_API_KEY = (
    os.getenv("HUGGINGFACE_API_KEY")
    or os.getenv("HF_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
)
FUTUREHOUSE_API_KEY = os.getenv("FUTUREHOUSE_API_KEY")
QWEN_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# Log only whether an API key is configured; never log key material.
if QWEN_API_KEY:
    logging.info("Qwen API key loaded from environment on module import.")
else:
    logging.warning("No API key found in environment on module import. Will check again at runtime.")


def _lazy_import_edison():
    """Lazy import of edison_client to speed up module loading"""
    global _edison_client, _edison_models
    if _edison_client is None:
        from edison_client import EdisonClient, JobNames
        from edison_client.models import TaskRequest, RuntimeConfig
        _edison_client = (EdisonClient, JobNames)
        _edison_models = (TaskRequest, RuntimeConfig)
    return _edison_client[0], _edison_client[1], _edison_models[0], _edison_models[1]


def _lazy_import_instructions():
    """Lazy import of instruction constants"""
    from app.tools.instruct import (
        CLARIFY_QUESTION_INSTRUCTIONS,      # 0
        SOCRATIC_PASS_INSTRUCTIONS,         # 1
        SOCRATIC_ANSWER_INSTRUCTIONS,       # 2
        TOT_INSTRUCTIONS,                   # 3
        RETRY_THINKING_INSTRUCTIONS,        # 4
        HYPOTHESIS_SYNTHESIS,               # 5
        HYPOTHESIS_ANALYSIS_REPORT,         # 6
        EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS, # 7
    )
    return (
        CLARIFY_QUESTION_INSTRUCTIONS,      # 0
        SOCRATIC_PASS_INSTRUCTIONS,         # 1
        SOCRATIC_ANSWER_INSTRUCTIONS,       # 2
        TOT_INSTRUCTIONS,                   # 3
        RETRY_THINKING_INSTRUCTIONS,        # 4
        HYPOTHESIS_SYNTHESIS,               # 5
        HYPOTHESIS_ANALYSIS_REPORT,         # 6
        EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS, # 7
    )


def _lazy_import_given_vars():
    """Lazy import of given_variables module"""
    global _given_vars
    if _given_vars is None:
        import app.tools.given_variables as given_vars
        _given_vars = given_vars
    return _given_vars


_EVIDENCE_ANCHOR_RE = re.compile(r"(?is)\s*Evidence\s+anchor\s*:.*$")
_EVIDENCE_CLAIM_RE = re.compile(r"\[(?:E|B|KG)\d+\]", re.IGNORECASE)
_EVIDENCE_DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE
)
_MONOVALENT_SPACER_RE = re.compile(
    r"\b(?:PEA|CHA)\b|\b(?:phenethylammonium|cyclohexylammonium)\b",
    re.IGNORECASE,
)
_DJ_TOPOLOGY_RE = re.compile(
    r"\bDion[- ]Jacobson\b|\bDJ(?:-type|\s+(?:phase|structure|spacer))\b",
    re.IGNORECASE,
)
_DIAMMONIUM_REQUEST_RE = re.compile(
    r"\b(?:diammonium|diamine|dicationic|dication|divalent)\b", re.IGNORECASE
)
_KNOWN_MONOCATION_DJ_CANDIDATES = (
    (
        re.compile(r"\bN-methyl-2-aminoethanol\b|\bNMEA(?:\+|\u207a)?\b", re.IGNORECASE),
        "N-methyl-2-aminoethanol has one protonatable amine and cannot supply the two "
        "ammonium termini required for a diammonium DJ spacer.",
    ),
    (
        re.compile(
            r"\b4-fluoro(?:phenethylammonium|phenethylamine)\b|\b4-FPEA(?:\+|\u207a)?\b|"
            r"\bphenethylammonium\b|\bPEA(?:\+|\u207a)?\b",
            re.IGNORECASE,
        ),
        "4-fluorophenethylammonium/phenethylammonium is a monovalent phenethylammonium "
        "spacer rather than a diammonium DJ linker.",
    ),
    (
        re.compile(r"\bcyclohexylammonium\b|\bCHA(?:\+|\u207a)?\b", re.IGNORECASE),
        "Cyclohexylammonium is monovalent and does not provide the two ammonium termini "
        "required for a diammonium DJ linker.",
    ),
)
_EXPLICIT_CHEMICAL_STRUCTURE_RE = re.compile(
    r"\b(?:SMILES|InChI|molecular formula|structural formula|explicit structure)\b",
    re.IGNORECASE,
)
_EXPLICIT_TWO_SITES_RE = re.compile(
    r"\b(?:diammonium|diamine|two\s+(?:protonatable\s+)?(?:amines?|nitrogens?|sites?)|"
    r"two\s+ammonium\s+termini|bis\s*\(\s*amino)",
    re.IGNORECASE,
)
_EXPLICIT_DICATION_RE = re.compile(
    r"(?:\b(?:net|intended|formal)?\s*2\s*\+(?!\w)|²⁺|\b(?:dication|diprotonated)\b)",
    re.IGNORECASE,
)


def _active_request_requires_diammonium_dj() -> bool:
    handoff = get_last_literature_handoff() or {}
    request = str(handoff.get("query") or "")
    return bool(
        _DIAMMONIUM_REQUEST_RE.search(request)
        and _DJ_TOPOLOGY_RE.search(request)
    )


def _guard_initial_diammonium_candidate(thought: str) -> tuple[str, bool]:
    """Reject known monocations when the active request explicitly requires a DJ dication."""
    if not _active_request_requires_diammonium_dj():
        return str(thought or "").strip(), False
    text = str(thought or "").strip()
    for pattern, reason in _KNOWN_MONOCATION_DJ_CANDIDATES:
        if pattern.search(text):
            return (
                "Candidate excluded from the requested diammonium DJ set: "
                f"{reason} It may be retained only as an explicitly monovalent RP/passivation "
                "comparator. Replace it with a chemically defined compound having two verified "
                "protonatable amines, an intended diprotonated 2+ state, and an explicit structure "
                "or SMILES before evaluating DJ formation.",
                True,
            )
    return text, False


def _guard_missing_dj_candidate_identity(thought: str) -> tuple[str, bool]:
    """Withhold topology-specific branches that never identify a candidate."""
    text = str(thought or "").strip()
    if not _active_request_requires_diammonium_dj():
        return text, False
    has_identity_basis = bool(
        _EXPLICIT_CHEMICAL_STRUCTURE_RE.search(text)
        or _EXPLICIT_TWO_SITES_RE.search(text)
        or _EXPLICIT_DICATION_RE.search(text)
    )
    if has_identity_basis:
        return text, False
    probe = re.sub(
        r"(?is)^\s*\**(?:exploratory\s+hypothesis|novel\s+proposal|unverified\s+proposal)"
        r"[^:]{0,220}:\**\s*",
        "",
        text,
    ).lstrip("* _-")
    unnamed_anaphora = re.match(
        r"(?is)^(?:it|its|this|the)\s+(?:compact\s+shape|candidate|spacer|molecule|compound|"
        r"proposal|structure|geometry)\b|^(?:it|its)\b",
        probe,
    )
    if not unnamed_anaphora:
        # A specific novel name can remain provisional. The downstream identity
        # requirement will expose its missing structure, sites, and/or charge.
        return text, False
    return (
        "Candidate definition required: this branch did not name a chemically identifiable "
        "diamine or dication, so it cannot be evaluated as a diammonium DJ spacer. Regenerate "
        "the option with a systematic chemical name, SMILES or molecular formula, two verified "
        "protonatable sites, the intended diprotonated net 2+ state, and a commercial source. "
        "Treat this branch as withheld rather than as a candidate.",
        True,
    )


def _append_dj_identity_requirement(thought: str) -> str:
    """Keep novel candidates, but expose missing structure/site/charge declarations."""
    text = str(thought or "").strip()
    if not _active_request_requires_diammonium_dj():
        return text
    missing: list[str] = []
    if not _EXPLICIT_CHEMICAL_STRUCTURE_RE.search(text):
        missing.append("a SMILES or molecular formula")
    if not _EXPLICIT_TWO_SITES_RE.search(text):
        missing.append("two verified protonatable amines")
    if not _EXPLICIT_DICATION_RE.search(text):
        missing.append("the intended diprotonated net 2+ state")
    if not missing:
        return text
    return (
        f"{text} Identity requirement before DJ assignment: provide "
        + ", ".join(missing)
        + "."
    )


def _strip_unverified_evidence_markers(thought: str) -> str:
    """Remove an incomplete evidence anchor before labeling a thought unverified."""
    cleaned = _EVIDENCE_ANCHOR_RE.sub("", str(thought or "")).strip()
    cleaned = _EVIDENCE_CLAIM_RE.sub("", cleaned)
    cleaned = _EVIDENCE_DOI_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,;:-")

    if _MONOVALENT_SPACER_RE.search(cleaned) and _DJ_TOPOLOGY_RE.search(cleaned):
        return (
            "Compare a chemically defined monovalent ammonium RP spacer with a "
            "chemically defined divalent ammonium DJ spacer while controlling the "
            "inorganic composition, layer number, processing, and interface position. "
            "Test whether spacer connectivity changes defect passivation, ion migration, "
            "and operational stability, treating every directional effect as an "
            "unverified prediction."
        )
    return cleaned


_DIRECT_EVIDENCE_COUNT_RE = re.compile(
    r"Packet inventory:\s*(\d+)\s+direct claim", re.IGNORECASE
)
_CONTINUATION_EVIDENCE_SECTION_RE = re.compile(
    r"(?is)\s*\**Evidence\s+anchor\s*:\**.*$"
)
_FORMAL_CHARGE_MUTATION_RE = re.compile(
    r"(?is)\b(?:alter|modify|tweak|increase|decrease|reduce|change)\w*\b.{0,100}"
    r"\b(?:divalenc\w*|formal charge|charge state)\b|"
    r"\b(?:divalenc\w*|formal charge|charge state)\b.{0,100}"
    r"\b(?:alter|modify|tweak|increase|decrease|reduce|change)\w*\b"
)


def _direct_evidence_count(evidence_context: str) -> int:
    text = str(evidence_context or "")
    match = _DIRECT_EVIDENCE_COUNT_RE.search(text)
    if match:
        return int(match.group(1))
    return len(set(re.findall(r"(?m)^\[(E\d+)\]", text, flags=re.IGNORECASE)))


def _selected_continuation_subject(selected_option: str) -> str:
    excluded = {
        "RP", "DJ", "PCE", "PL", "XRD", "AFM", "FTIR", "TOF", "SIMS",
        "FAP", "LED", "UV", "VIS",
    }
    candidates = re.findall(
        r"\b[A-Z]{2,8}(?:[\u00b2\u00b3]?\u207a|\d*\+)?",
        str(selected_option or ""),
    )
    for candidate in candidates:
        base = re.sub(r"[^A-Z]", "", candidate.upper())
        if base and base not in excluded:
            return candidate
    return "selected"


def _safe_ungrounded_continuations(selected_option: str) -> list[str]:
    subject = _selected_continuation_subject(selected_option)
    label = f"the selected {subject} proposal" if subject != "selected" else "the selected proposal"
    return [
        (
            f"Unverified continuation (no direct supporting evidence retrieved): Does {label} "
            "produce the claimed effect relative to matched controls, and which direct "
            "measurement would falsify that effect?"
        ),
        (
            f"Unverified continuation (no direct supporting evidence retrieved): Which "
            f"measurement can distinguish the proposed mechanism of {label} from competing "
            "explanations while composition, processing, and geometry are held constant?"
        ),
        (
            f"Unverified continuation (no direct supporting evidence retrieved): Which fixed "
            f"structural variables of {label}, such as linker length, rigidity, substitution, "
            "or packing, can be varied while keeping chemical identity and formal charge "
            "explicit and controlled?"
        ),
    ]


def _selected_entity_terms(selected_option: str) -> set[str]:
    text = str(selected_option or "")
    terms = {
        token.lower()
        for token in re.findall(
            r"\b([A-Z]{2,8})(?:[\u00b2\u00b3]?\u207a|\d*\+)?",
            text,
        )
    }
    terms -= {"rp", "dj", "pce", "pl", "xrd", "afm", "ftir", "tof", "sims", "fap"}
    for match in re.findall(
        r"([A-Za-z0-9,\-]+(?:amine|ammonium|diammonium|dication|cation))",
        text,
        flags=re.IGNORECASE,
    ):
        normalized = re.sub(r"[^a-z0-9]", "", match.lower())
        if len(normalized) >= 4:
            terms.add(normalized)
    return {term for term in terms if len(term) >= 3}


def _evidence_claim_blocks(evidence_context: str) -> dict[str, str]:
    text = str(evidence_context or "")
    matches = list(re.finditer(r"(?m)^\[(E\d+)\]", text, flags=re.IGNORECASE))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1).upper()] = text[match.start():end]
    return blocks


def _property_terms(text: str) -> set[str]:
    lowered = str(text or "").lower()
    vocabulary = {
        "ion migration", "charge transport", "mobility", "band alignment", "bandgap",
        "recombination", "defect", "stability", "bonding", "carrier lifetime",
        "photoluminescence", "crystallinity", "phase stability", "efficiency",
    }
    return {term for term in vocabulary if term in lowered}


def _sanitize_ungrounded_continuation(option: str, fallback: str) -> str:
    text = str(option or "").strip()
    if _FORMAL_CHARGE_MUTATION_RE.search(text):
        return fallback
    text = _CONTINUATION_EVIDENCE_SECTION_RE.sub("", text)
    text = _EVIDENCE_CLAIM_RE.sub("", text)
    text = _EVIDENCE_DOI_RE.sub("", text)
    text = re.sub(
        r"(?i)^Why\s+is\s+(.+?)\s+expected\s+to\s+",
        r"Does \1 ",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text).strip(" ,;:-")
    if not text:
        return fallback
    if not text.lower().startswith(("unverified continuation", "novel proposal")):
        text = "Unverified continuation (no direct supporting evidence retrieved): " + text
    return text


def _gate_continuation_options(
    options: list[str], selected_option: str, evidence_context: str
) -> list[str]:
    fallbacks = _safe_ungrounded_continuations(selected_option)
    direct_count = _direct_evidence_count(evidence_context)
    if direct_count <= 0:
        logging.warning(
            "Continuation evidence gate: no direct claims; removing all support anchors."
        )
        return fallbacks

    entity_terms = _selected_entity_terms(selected_option)
    claim_blocks = _evidence_claim_blocks(evidence_context)
    gated: list[str] = []
    for index, option in enumerate(list(options)[:3]):
        text = str(option or "").strip()
        invalid = bool(_FORMAL_CHARGE_MUTATION_RE.search(text))
        anchors = re.findall(r"\[((?:E|B|KG)\d+)\]", text, flags=re.IGNORECASE)
        if any(not anchor.upper().startswith("E") for anchor in anchors):
            invalid = True
        for anchor in [item.upper() for item in anchors if item.upper().startswith("E")]:
            block = claim_blocks.get(anchor, "")
            normalized_block = re.sub(r"[^a-z0-9]", "", block.lower())
            if not block or (entity_terms and not any(term in normalized_block for term in entity_terms)):
                invalid = True
                break
            option_properties = _property_terms(text)
            block_properties = _property_terms(block)
            if option_properties and not option_properties.intersection(block_properties):
                invalid = True
                break
        gated.append(fallbacks[index] if invalid else text)
    while len(gated) < 3:
        gated.append(fallbacks[len(gated)])
    return gated[:3]


def get_local_literature_context(
    query: str,
    limit: int = 5,
    *,
    require_evidence: bool = False,
    require_direct_support: bool = False,
    retrieval_attempts: int = 3,
) -> str:
    """Retrieve local evidence and trigger a small literature job if evidence is thin."""
    try:
        from app.tools.polaris_orchestrator import PolarisOrchestrator
        packet = PolarisOrchestrator().literature_evidence(
            query,
            limit=limit,
            max_chars=3000,
            auto_expand_on_miss=True,
            min_items=1,
            retrieval_max_papers=3,
            retrieval_timeout_seconds=120,
            retrieval_attempts=retrieval_attempts,
            require_strong_evidence=require_direct_support,
        )
        provenance = dict(packet.get("provenance") or {})
        _last_literature_handoff.set({
            "query": query,
            "strong_evidence_count": int(provenance.get("strong_evidence_count") or 0),
            "evidence_items_used_for_gate": int(provenance.get("evidence_items_used_for_gate") or 0),
            "grounding_status": provenance.get("grounding_status"),
            "grounding_warning": provenance.get("grounding_warning"),
            "retrieval_attempt_count": int(provenance.get("retrieval_attempt_count") or 0),
            "retrieval_attempts": provenance.get("retrieval_attempts") or [],
            "claim_count": int(provenance.get("claim_count") or 0),
            "background_claim_count": int(provenance.get("background_claim_count") or 0),
            "claim_ids": provenance.get("claim_ids") or [],
            "source_ids": provenance.get("source_ids") or [],
            "controller_packet_path": provenance.get("controller_packet_path"),
            "background_retrieval_job": provenance.get("background_retrieval_job"),
            "background_retrieval_error": provenance.get("background_retrieval_error"),
            "retrieval_query": provenance.get("retrieval_query") or query,
            "evidence_gate_requirement": provenance.get("evidence_gate_requirement"),
            "fresh_retrieval_required": bool(provenance.get("fresh_retrieval_required")),
            "fresh_retrieval_performed": bool(provenance.get("fresh_retrieval_performed")),
            "formatted_context": str(packet.get("formatted_context") or ""),
        })
        return packet.get("formatted_context", "Local LiteratureAgent evidence was unavailable.")
    except Exception as exc:
        _last_literature_handoff.set({"query": query, "error": repr(exc)})
        logging.warning(f"Local LiteratureAgent evidence retrieval unavailable: {exc}")
        if require_evidence:
            raise LiteratureEvidenceUnavailableError(str(exc)) from exc
        return "Local LiteratureAgent evidence was unavailable for this analysis."


def get_last_literature_handoff() -> dict | None:
    """Return request-local LiteratureAgent provenance for deterministic UI reporting."""
    value = _last_literature_handoff.get()
    return dict(value) if isinstance(value, dict) else None


def set_last_literature_handoff(value: dict | None) -> None:
    """Restore the session-persisted handoff at the start of a new API request."""
    _last_literature_handoff.set(dict(value) if isinstance(value, dict) else None)


def _normalized_evidence_scope(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _reuse_selected_branch_context(selected_scope: str) -> str:
    handoff = get_last_literature_handoff() or {}
    context = str(handoff.get("formatted_context") or "").strip()
    prior_scope = _normalized_evidence_scope(
        str(
            handoff.get("selected_scope")
            or handoff.get("retrieval_query")
            or handoff.get("query")
            or ""
        )
    )
    selected = _normalized_evidence_scope(selected_scope)
    if not context or not selected or not prior_scope:
        return ""
    if selected not in prior_scope and prior_scope not in selected:
        return ""
    handoff["used_for_hypothesis_synthesis"] = True
    handoff["synthesis_reused_selected_branch_packet"] = True
    _last_literature_handoff.set(handoff)
    return context


def _mark_synthesis_evidence_context(context: str) -> None:
    handoff = get_last_literature_handoff() or {}
    handoff["formatted_context"] = str(context or "")
    handoff["used_for_hypothesis_synthesis"] = True
    handoff.setdefault("synthesis_reused_selected_branch_packet", False)
    _last_literature_handoff.set(handoff)


def _merge_evidence_contexts(literature_context: str, *grounded_texts: str) -> str:
    contexts = [str(literature_context or "").strip()]
    for value in grounded_texts:
        text = str(value or "")
        marker = text.find("LITERATURE_KG_CONTEXT")
        if marker < 0:
            continue
        kg_context = text[marker:]
        boundary = kg_context.find("\n\nNOVELTY_BOUNDARY")
        if boundary >= 0:
            kg_context = kg_context[:boundary]
        kg_context = kg_context.strip()
        if kg_context and kg_context not in contexts:
            contexts.append(kg_context)
    return "\n\n".join(item for item in contexts if item)


def _audit_and_store_literature_use(response: str, literature_context: str) -> dict:
    claim_ids = list(
        dict.fromkeys(
            re.findall(r"\[((?:E|B|KG)\d+)\]", literature_context, flags=re.IGNORECASE)
        )
    )
    source_candidates = re.findall(r"source=([^\n|]+)", literature_context)
    source_candidates.extend(
        re.findall(r"(?im)^\s*DOI:\s*([^\s|)]+)", literature_context)
    )
    source_ids = list(
        dict.fromkeys(item.strip().rstrip(".,;") for item in source_candidates if item.strip())
    )
    response_lower = str(response or "").lower()
    used_claim_ids = [claim_id for claim_id in claim_ids if claim_id.lower() in response_lower]
    used_source_ids = [source_id for source_id in source_ids if source_id.lower() in response_lower]
    citation_complete = (
        (not claim_ids or bool(used_claim_ids))
        and (not source_ids or bool(used_source_ids))
    )
    audit = {
        "available_claim_ids": claim_ids,
        "available_source_ids": source_ids,
        "used_claim_ids": used_claim_ids,
        "used_source_ids": used_source_ids,
        "evidence_packet_used": citation_complete,
        "citation_complete": citation_complete,
    }
    current = get_last_literature_handoff() or {}
    current.update(audit)
    _last_literature_handoff.set(current)
    return audit


def generate_text_with_llm(prompt: str) -> str:
    """Generate text using provider/model/API key from Settings (SQLite), not stale .env."""
    try:
        from app.services.llm_runtime import require_api_key
        from app.tools.llm_client import generate_text as llm_generate

        cfg = require_api_key()
        logging.info(
            "Using %s model %s, API key present: %s (source: settings DB)",
            cfg.provider,
            cfg.model,
            bool(cfg.api_key),
        )
        text = llm_generate(
            prompt=prompt,
            api_key=cfg.api_key,
            provider=cfg.provider,
            model=cfg.model,
            qwen_base_url=cfg.qwen_base_url,
        )
        logging.info(f"Successfully generated text (length: {len(text)})")
        return text
    except ValueError as e:
        logging.error(f"API key or response error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise

def clarify_question(question: str, reasoning_policy: str = "auto") -> str:
    """Clarifying question given by user """
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        CLARIFY_QUESTION_INSTRUCTIONS = instructions[0]

        #Creating prompt for clarified question
        clarified_question_prompt = f"""
        {CLARIFY_QUESTION_INSTRUCTIONS}

        Question: {question}

        Clarify the user's question without adding literature findings, named examples,
        calculations, or mechanisms that were not in the question. Put uncertain premises
        under assumptions; do not turn them into factual claims. Preserve specialized terms
        from the question verbatim instead of inventing definitions for them. If a definition
        is needed but was not supplied, label it as an unresolved definition.
        """
        if reasoning_policy and reasoning_policy.strip().lower() not in ("none", "off", "disabled"):
            clarified_question_prompt, _reasoning_meta = _apply_reasoning_policy(
                clarified_question_prompt,
                task=f"Clarify research question: {question[:200]}",
                task_type="hypothesis",
                policy=reasoning_policy,
            )
            _log_reasoning_metadata(_reasoning_meta, fn_name="clarify_question")
        llm_response = generate_text_with_llm(clarified_question_prompt)
        if llm_response is None or not llm_response.strip():
            logging.error("LLM returned empty response for clarify_question")
            return None
        return llm_response

    except ValueError as e:
        logging.error(f"API key error in clarify_question: {e}")
        raise
    except Exception as e:
        import traceback
        logging.error(f"LLM clarifying question failed ({type(e).__name__}): {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Clarify question failed: {e}") from e


def socratic_pass(clarified_question_response: str, reasoning_policy: str = "auto") -> str:
    """LLM is performing self questioning on clarified question response"""

    try:
        # Lazy import instructions and given_vars
        instructions = _lazy_import_instructions()
        SOCRATIC_PASS_INSTRUCTIONS = instructions[1]
        given_vars = _lazy_import_given_vars()

        socratic_pass_prompt = f"""
        {SOCRATIC_PASS_INSTRUCTIONS}
        User Question: {clarified_question_response}
        Socratic Principles: {given_vars.socratic_principles}
        """

        if reasoning_policy and reasoning_policy.strip().lower() not in ("none", "off", "disabled"):
            socratic_pass_prompt, _reasoning_meta = _apply_reasoning_policy(
                socratic_pass_prompt,
                task=f"Apply probing questions to: {clarified_question_response[:300]}",
                task_type="gap_analysis",
                policy=reasoning_policy,
            )
            _log_reasoning_metadata(_reasoning_meta, fn_name="socratic_pass")

        llm_response = generate_text_with_llm(socratic_pass_prompt)
        return llm_response

    except Exception as e:
        logging.error(f"SOCRATIC pass failed: {e}")


def socratic_pass_and_tot(
    clarified_question: str,
    reasoning_policy: str = "auto",
    evidence_query: str | None = None,
) -> tuple:
    """
    Single LLM call: socratic probing + three lines of thought.
    Falls back to separate socratic_pass + tot_generation if parsing fails.
    """
    try:
        instructions = _lazy_import_instructions()
        SOCRATIC_PASS_INSTRUCTIONS = instructions[1]
        TOT_INSTRUCTIONS = instructions[3]
        given_vars = _lazy_import_given_vars()
        literature_context = get_local_literature_context(
            evidence_query or clarified_question,
            limit=2,
            require_evidence=False,
        )

        combined_prompt = f"""
Complete BOTH steps in this ONE response.

STEP 1 — Socratic pass (probing questions with brief reasoning):
{SOCRATIC_PASS_INSTRUCTIONS}
User Question: {clarified_question}
Socratic Principles: {given_vars.socratic_principles}

STEP 2 — Three distinct lines of thought (use Step 1 as reasoning; do NOT output a hypothesis report):
{TOT_INSTRUCTIONS}
User Question: {clarified_question}
Socratic Answers: Use your Step 1 probing questions and concise self-answers inline.

Literature evidence contract:
{literature_context}

Keep the original hypothesis-agent behavior: produce useful Socratic questions
and three scientifically distinct exploratory lines of thought. When the packet
contains numbered evidence claims, every thought that draws on literature MUST
include an `Evidence anchor:` naming at least one claim ID plus its DOI or
paper_slug. Explain what the source supports and what remains inference. If a
thought is not directly supported by the packet, label it clearly as an
exploratory or novel proposal rather than implying literature support.

For partial-context evidence, preserve the variable and experimental scope the
source actually tested and add an `Evidence boundary:` stating what it did not
test. Never attach a source to a different compound, mechanism, property, or
relationship merely because it concerns the same broad material class. Do not
invent citations, reported results, calculations, or numerical effect sizes.

Define every proposed chemical identity unambiguously. In standard layered
lead-halide perovskites, monovalent ammonium spacers are RP-type; a DJ assignment
requires an appropriate divalent spacer linking adjacent inorganic slabs. Novel
entities remain allowed, but must be labeled `Novel proposal - not supported by
the retrieved packet` when absent from both the question and packet.

Required format:
## Socratic Pass
<probing questions with reasoning>

## Distinct Line of Thought 1:
<first thought>

## Distinct Line of Thought 2:
<second thought>

## Distinct Line of Thought 3:
<third thought>
"""
        if reasoning_policy and reasoning_policy.strip().lower() not in ("none", "off", "disabled"):
            combined_prompt, _reasoning_meta = _apply_reasoning_policy(
                combined_prompt,
                task=f"Apply probing questions and generate exploration paths for: {clarified_question[:200]}",
                task_type="gap_analysis",
                policy=reasoning_policy,
            )
            _log_reasoning_metadata(_reasoning_meta, fn_name="socratic_pass_and_tot")

        llm_response = generate_text_with_llm(combined_prompt)
        if not llm_response or not str(llm_response).strip():
            raise ValueError("empty combined socratic/tot response")

        use_audit = _audit_and_store_literature_use(
            str(llm_response), literature_context
        )
        partial_context = "partial_context_only" in literature_context
        if (
            (use_audit["available_claim_ids"] and not use_audit["citation_complete"])
            or partial_context
        ):
            repair_prompt = f"""
Revise the response below without changing its section structure or core ideas.
For each line of thought that uses literature, add an `Evidence anchor:` with a
directly aligned [E#] claim and its DOI or paper_slug. Explain the evidence
boundary and keep unsupported extensions explicitly labeled as novel proposals.
Do not use [B#] or [KG#] as direct validation, and do not add facts, materials,
results, or citations absent from the packet.

LiteratureAgent packet:
{literature_context}

Response to revise:
{llm_response}
"""
            revised = generate_text_with_llm(repair_prompt)
            if revised and str(revised).strip():
                llm_response = revised
            _audit_and_store_literature_use(str(llm_response), literature_context)

        text = str(llm_response)
        distinct_match = re.search(r"Distinct\s+Line\s+of\s+Thought", text, re.IGNORECASE)
        if distinct_match:
            soc_pass = re.sub(
                r"^##\s*Socratic\s+Pass\s*:?\s*",
                "",
                text[: distinct_match.start()],
                flags=re.IGNORECASE,
            ).strip()
        else:
            soc_pass = re.sub(
                r"^##\s*Socratic\s+Pass\s*:?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
        thoughts = _finalize_initial_thoughts(
            _parse_tot_thoughts_from_text(text), literature_context
        )

        substantial = [t for t in thoughts if t and len(str(t).strip()) > 20]
        if not soc_pass or len(substantial) < 3:
            logging.info("Combined socratic/tot parse weak; falling back to sequential calls")
            soc_pass = socratic_pass(clarified_question, reasoning_policy=reasoning_policy) or ""
            thoughts = tot_generation(soc_pass, clarified_question, None, reasoning_policy=reasoning_policy)
        return soc_pass, thoughts
    except LiteratureEvidenceUnavailableError:
        raise
    except Exception as e:
        logging.warning(f"socratic_pass_and_tot failed, using sequential flow: {e}")
        soc_pass = socratic_pass(clarified_question, reasoning_policy=reasoning_policy) or ""
        return soc_pass, tot_generation(soc_pass, clarified_question, None, reasoning_policy=reasoning_policy)


def _mark_unverified_exploration(thought: object) -> str:
    """Prevent weakly grounded exploration from sounding literature-confirmed."""
    text = str(thought or "").strip()
    if not text:
        return text
    attribution_markers = (
        "supported by", "as evidenced", "studies show", "studies demonstrate",
        "literature shows", "literature indicates", "prior literature", "reported studies",
        "calculations showing", "calculations demonstrate",
    )
    kept_sentences = []
    removed_attribution = False
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lowered = sentence.lower()
        has_source_id = bool(re.search(r"\b10\.\d{4,9}/\S+|paper_slug", sentence, re.IGNORECASE))
        if any(marker in lowered for marker in attribution_markers) and not has_source_id:
            removed_attribution = True
            continue
        kept_sentences.append(sentence)
    cleaned = " ".join(kept_sentences).strip() or "This possibility requires direct literature or experimental testing."
    prefix = "Exploratory hypothesis (not directly supported by the current local packet): "
    if removed_attribution:
        cleaned += " Any implied prior support was omitted because no DOI or paper_slug was supplied."
    return prefix + cleaned


def _thought_has_complete_direct_anchor(thought: str, literature_context: str) -> bool:
    """Require a packet [E#] identifier and that claim's source in the same thought."""
    text = str(thought or "")
    blocks = _evidence_claim_blocks(literature_context)
    anchors = {
        item.upper()
        for item in re.findall(r"\[(E\d+)\]", text, flags=re.IGNORECASE)
    }
    lowered = text.lower()
    for anchor in anchors:
        block = blocks.get(anchor, "")
        sources = re.findall(r"source=([^\n|]+)", block, flags=re.IGNORECASE)
        sources.extend(
            re.findall(r"(?im)^\s*DOI:\s*([^\s|)]+)", block)
        )
        if any(source.strip().rstrip(".,;").lower() in lowered for source in sources):
            return True
    return False


def _finalize_initial_thoughts(
    thoughts: list[object], literature_context: str
) -> list[str]:
    """Make evidence visibility deterministic for every initial thought."""
    handoff = get_last_literature_handoff() or {}
    direct_count = max(
        int(handoff.get("strong_evidence_count") or 0),
        _direct_evidence_count(literature_context),
    )
    finalized: list[str] = []
    for value in list(thoughts)[:3]:
        thought = str(value or "").strip()
        if not thought:
            finalized.append("")
            continue
        thought, topology_rejected = _guard_initial_diammonium_candidate(thought)
        if topology_rejected:
            finalized.append(thought)
            continue
        thought, identity_withheld = _guard_missing_dj_candidate_identity(thought)
        if identity_withheld:
            finalized.append(thought)
            continue
        thought = _append_dj_identity_requirement(thought)
        already_unverified = thought.lower().startswith(
            ("exploratory hypothesis", "novel proposal", "unverified")
        )
        if direct_count > 0 and _thought_has_complete_direct_anchor(
            thought, literature_context
        ):
            finalized.append(thought)
        elif already_unverified:
            finalized.append(thought)
        else:
            cleaned = _strip_unverified_evidence_markers(thought)
            finalized.append(_mark_unverified_exploration(cleaned))
    while len(finalized) < 3:
        finalized.append("")
    _audit_and_store_literature_use("\n\n".join(finalized), literature_context)
    return finalized[:3]


def _parse_tot_thoughts_from_text(text: str) -> list:
    """Lightweight TOT parser for combined socratic+tot responses."""
    if not text or not str(text).strip():
        return ["", "", ""]
    pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    thoughts: list = []
    if len(matches) >= 1:
        for i, match in enumerate(matches):
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start_pos:end_pos].strip()
            if chunk and len(chunk) > 10:
                thoughts.append(chunk)
    if len(thoughts) < 3:
        numbered = re.findall(r"^\s*\d+\.\s+(.+)$", text, flags=re.MULTILINE)
        thoughts = [t.strip() for t in numbered if len(t.strip()) > 20][:3]
    while len(thoughts) < 3:
        thoughts.append("")
    return thoughts[:3]


def socratic_answer_questions(clarified_question: str, probing_questions: str, reasoning_policy: str = "auto") -> str:
    """LLM answers the probing questions it previously asked itself, building deeper reasoning"""
    print("\n" + "="*80)
    print("DEBUG: INSIDE socratic_answer_questions")
    print(f"  clarified_question length: {len(clarified_question)}")
    print(f"  probing_questions length: {len(probing_questions)}")
    print("="*80 + "\n")
    try:
        # Pre-process probing questions to remove "Reasoning:" sections
        # This helps ensure the LLM focuses on answering the questions, not explaining why they were asked
        # Split by question patterns first, then clean each question
        import re
        
        # Pattern to match questions (lines that end with "?" and don't start with "Reasoning")
        question_pattern = r'^[^R].*\?'
        
        # Split into blocks - each question followed by its reasoning
        blocks = re.split(r'\n(?=\w)', probing_questions)  # Split on newline followed by word char
        
        cleaned_questions = []
        for block in blocks:
            lines = block.split('\n')
            question_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Skip reasoning lines
                if line_stripped.lower().startswith('reasoning:'):
                    break  # Stop at reasoning, don't include it
                # If line contains "Reasoning:" in middle, take only before it
                if 'Reasoning:' in line_stripped:
                    question_part = line_stripped.split('Reasoning:')[0].strip()
                    if question_part:
                        question_lines.append(question_part)
                    break  # Stop after this line
                # Add question lines
                if line_stripped and not line_stripped.lower().startswith('reasoning'):
                    question_lines.append(line_stripped)
            
            if question_lines:
                cleaned_questions.append('\n'.join(question_lines))
        
        # Join back together
        cleaned_questions_text = '\n\n'.join(cleaned_questions)
        
        # If we removed everything, fall back to original (shouldn't happen, but safety check)
        if not cleaned_questions_text.strip():
            cleaned_questions_text = probing_questions
            logging.warning("Could not clean probing questions, using original")
        else:
            logging.info(f"Cleaned probing questions (removed Reasoning sections)")
            logging.info(f"  Original length: {len(probing_questions)}, Cleaned length: {len(cleaned_questions_text)}")
            logging.info(f"  First 300 chars of cleaned: {cleaned_questions_text[:300]}...")
        
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        SOCRATIC_ANSWER_INSTRUCTIONS = instructions[2]  # SOCRATIC_ANSWER_INSTRUCTIONS is now at index 2

        literature_context = get_local_literature_context(
            f"{clarified_question}\n{cleaned_questions_text}",
            limit=4,
        )
        socratic_answer_prompt = f"""
{SOCRATIC_ANSWER_INSTRUCTIONS}

User Question: {clarified_question}

Probing Questions: {cleaned_questions_text}

Evidence-grounded local literature packet:
{literature_context}

Use this packet to ground examples and mechanisms. Clearly distinguish direct
literature evidence from scientific inference. Do not invent citations or metrics.
"""
        
        logging.info("Constructed socratic answer prompt")
        logging.info(f"  Prompt length: {len(socratic_answer_prompt)}")
        logging.info(f"  Instructions length: {len(SOCRATIC_ANSWER_INSTRUCTIONS)}")
        logging.info(f"  First 300 chars of full prompt: {socratic_answer_prompt[:300]}...")
        
        logging.info("Generating answers to socratic questions...")
        logging.info(f"  Clarified question length: {len(clarified_question)}")
        logging.info(f"  Probing questions length: {len(probing_questions)}")
        logging.info(f"  Cleaned questions length: {len(cleaned_questions_text)}")
        logging.info(f"  Cleaned questions preview: {cleaned_questions_text[:500]}...")

        try:
            _wrapped = _generate_with_reasoning_wrapper(
                base_prompt=socratic_answer_prompt,
                task=f"Answer Socratic probing questions for: {clarified_question[:200]}",
                task_type="hypothesis",
                policy=reasoning_policy,
                fn_name="socratic_answer_questions",
            )
            llm_response = _wrapped["response"]
            logging.info(f"LLM call completed successfully")
        except Exception as llm_error:
            logging.error(f"LLM call failed with error: {llm_error}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
        
        logging.info(f"Raw LLM response received (length: {len(llm_response) if llm_response else 0})")
        if llm_response:
            logging.info(f"  First 500 chars of raw response: {llm_response[:500]}...")
        else:
            logging.error("LLM returned None or empty response")
            logging.error("  This is unusual - the LLM should return something, even if it's an error message")
            return None
        
        if llm_response:
            # CRITICAL: Check if LLM generated a hypothesis report instead of answers
            if "Hypothesis Report" in llm_response or "Hypothesis Evaluation Report" in llm_response:
                logging.error("LLM generated a hypothesis report instead of socratic answers!")
                logging.error("  Full response (first 1000 chars): {0}".format(llm_response[:1000]))
                logging.error("  TEMPORARILY ALLOWING THIS THROUGH SO USER CAN SEE IT")
                # TEMP: Don't return None, let it through so user can see what's happening
                # return None
            
            # Check for novelty/plausibility/testability which indicates a hypothesis report
            if "Novelty:" in llm_response or "Plausibility:" in llm_response or "Testability:" in llm_response:
                logging.error("LLM response contains hypothesis report markers (Novelty/Plausibility/Testability)")
                logging.error("  Full response (first 1000 chars): {0}".format(llm_response[:1000]))
                logging.error("  TEMPORARILY ALLOWING THIS THROUGH SO USER CAN SEE IT")
                # TEMP: Don't return None, let it through so user can see what's happening
                # return None
            
            # Check if LLM is just restating reasoning instead of answering
            # Look for patterns that indicate it's explaining why questions were asked
            reasoning_patterns = [
                "this question probes",
                "this question explores",
                "this question investigates",
                "this question examines",
                "this question seeks",
                "this question addresses",
                "reasoning:",
                "the reasoning behind"
            ]
            
            response_lower = llm_response.lower()
            reasoning_count = sum(1 for pattern in reasoning_patterns if pattern in response_lower)
            
            if reasoning_count >= 3:  # If multiple reasoning patterns found, likely not answering
                logging.warning(f"LLM response contains {reasoning_count} reasoning patterns - may be explaining why questions were asked instead of answering")
                logging.warning(f"  First 500 chars: {llm_response[:500]}...")
                # Don't return None - let it through but log warning
                # The instructions should handle this, but we want to track it
                # Still return the response so user can see what happened
            
            # Additional validation: Check if response actually answers questions vs just restating reasoning
            # Look for actual answers (specific materials, mechanisms, examples)
            answer_indicators = [
                "include", "examples include", "such as", "specifically", 
                "functional groups", "molecular", "classes of", "types of",
                "coordination", "interactions", "binding", "properties"
            ]
            answer_count = sum(1 for indicator in answer_indicators if indicator.lower() in response_lower)
            
            if reasoning_count >= 3 and answer_count < 2:
                logging.error(f"LLM response appears to be restating reasoning instead of answering!")
                logging.error(f"  Reasoning patterns: {reasoning_count}, Answer indicators: {answer_count}")
                logging.error(f"  First 500 chars: {llm_response[:500]}...")
                # Still return it, but log the issue - the instructions should handle this
            
            logging.info(f"Generated socratic answers (length: {len(llm_response)})")
            logging.info(f"  First 500 chars: {llm_response[:500]}...")
            logging.info(f"  Contains reasoning patterns: {reasoning_count}, Answer indicators: {answer_count}")
            
            # Final validation - if it looks like it's just restating reasoning, log but still return
            if reasoning_count >= 3 and answer_count < 2:
                logging.warning("WARNING: Response may be restating reasoning instead of answering")
                logging.warning("  Reasoning patterns: {0}, Answer indicators: {1}".format(reasoning_count, answer_count))
                logging.warning("  BUT still returning it so user can see what was generated")
        else:
            logging.error("Empty response from socratic answer generation")
            logging.error("  This means socratic answers will not be displayed")
            logging.error("  Returning None - fallback will use probing questions directly")
            
        print("\n" + "="*80)
        print("DEBUG: RETURNING from socratic_answer_questions")
        print(f"  Return value is None: {llm_response is None}")
        print(f"  Return value type: {type(llm_response)}")
        if llm_response:
            print(f"  Return value length: {len(llm_response)}")
            print(f"  First 300 chars: {llm_response[:300]}")
        print("="*80 + "\n")
        
        logging.info("=" * 80)
        logging.info(f"RETURNING from socratic_answer_questions")
        logging.info(f"  Return value is None: {llm_response is None}")
        logging.info(f"  Return value type: {type(llm_response)}")
        if llm_response:
            logging.info(f"  Return value length: {len(llm_response)}")
        logging.info("=" * 80)
        
        return llm_response
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"DEBUG: EXCEPTION in socratic_answer_questions: {e}")
        print("="*80 + "\n")
        logging.error(f"SOCRATIC answer generation failed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def tot_generation_experimental_plan(socratic_pass_questioning: str, clarified_question: str, experimental_constraints: str) -> list:
    """LLM produces three distinct experimental plans given constraints"""
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS = instructions[7]

        tot_generation_prompt = f"""
        {EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS}
        User Question: {clarified_question}
        Probing Questions: {socratic_pass_questioning}
        Experimental Constraints: {experimental_constraints}
        """

        llm_response = generate_text_with_llm(tot_generation_prompt)

        if isinstance(llm_response, list):
            llm_response = " ".join(llm_response)
        elif llm_response is None:
            llm_response = ""

        # Parse experimental plans (look for "Experimental Plan" pattern)
        pattern = r"Experimental Plan\s*\d*\s*:\s*"
        raw_plans = re.split(pattern, llm_response, flags=re.IGNORECASE)

        plans = [p.strip() for p in raw_plans if p.strip()]

        # If no plans found, try alternative parsing
        if not plans or len(plans) == 0:
            numbered_pattern = r'^\d+\.?\s+'
            lines = llm_response.splitlines()
            for line in lines:
                line = line.strip()
                if line and (re.match(numbered_pattern, line) or line.startswith('-') or line.startswith('•')):
                    clean_plan = re.sub(numbered_pattern, '', line)
                    clean_plan = re.sub(r'^[-•]\s*', '', clean_plan)
                    if clean_plan and len(clean_plan) > 20:
                        plans.append(clean_plan)
            
            if not plans:
                sentences = re.split(r'(?<=[.!?])\s+', llm_response)
                plans = [s.strip() for s in sentences if len(s.strip()) > 30][:3]

        # Ensure we have at least 3 plans
        while len(plans) < 3:
            plans.append("")

        return plans[:3]

    except Exception as e:
        logging.error(f"Experimental plan TOT generation failed: {e}")
        return ["Error generating experimental plans", "Please check API key and try again", ""]


def tot_generation(socratic_pass_questioning: str, clarified_question: str, socratic_answers: str = None, reasoning_policy: str = "auto") -> list:
    """LLM produces three distinct lines of thought (rationale, predicted outcomes, and next step question)

    Args:
        socratic_pass_questioning: The probing questions generated
        clarified_question: The clarified user question
        socratic_answers: The answers to the probing questions (if available)
        reasoning_policy: The philosophical reasoning school to apply (default "auto")
    """
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        TOT_INSTRUCTIONS = instructions[3]  # TOT_INSTRUCTIONS is now at index 3 (after SOCRATIC_ANSWER_INSTRUCTIONS)

        # If we have answers, use them; otherwise use questions
        reasoning_context = socratic_answers if socratic_answers else socratic_pass_questioning
        literature_context = get_local_literature_context(
            clarified_question,
            limit=4,
            require_evidence=False,
        )

        tot_generation_prompt = f"""
        {TOT_INSTRUCTIONS}
        User Question: {clarified_question}
        Socratic Answers: {reasoning_context}
        Evidence-grounded local literature packet:
        {literature_context}

        Prefer lines of thought directly supported by the packet. Every supported
        statement must include an aligned [E#] evidence anchor plus its DOI or
        paper_slug. Treat [B#] and [KG#] as background or analogy only. Novel
        extrapolations remain allowed, but explicitly label them as unsupported
        proposals rather than established findings. Do not invent citations.
        """

        if socratic_answers:
            logging.info(f"Generating TOT with socratic answers (length: {len(socratic_answers)})...")
        else:
            logging.info(f"Generating TOT with probing questions (answers not available)...")

        _wrapped = _generate_with_reasoning_wrapper(
            base_prompt=tot_generation_prompt,
            task=f"Generate three exploration paths for: {clarified_question[:300]}",
            task_type="hypothesis",
            policy=reasoning_policy,
            fn_name="tot_generation",
        )
        llm_response = _wrapped["response"]

        if isinstance(llm_response, list):
            llm_response = " ".join(llm_response)  # combine if list
        elif llm_response is None:
            llm_response = ""  # avoid NoneType error
        
        # CRITICAL: Check if LLM generated a hypothesis report instead of thoughts
        if llm_response and ("Hypothesis Report" in llm_response or "Hypothesis Evaluation Report" in llm_response):
            logging.error("LLM generated a hypothesis report instead of TOT thoughts!")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []
        
        # Check for novelty/plausibility/testability which indicates a hypothesis report
        if llm_response and ("Novelty:" in llm_response or "Plausibility:" in llm_response or "Testability:" in llm_response):
            logging.error("LLM response contains hypothesis report markers (Novelty/Plausibility/Testability)")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []

        # CRITICAL: Check if LLM generated a hypothesis report instead of thoughts
        if llm_response and ("Hypothesis Report" in llm_response or "Hypothesis Evaluation Report" in llm_response):
            logging.error("LLM generated a hypothesis report instead of TOT thoughts!")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []

        # Check for novelty/plausibility/testability which indicates a hypothesis report
        if llm_response and (
                "Novelty:" in llm_response or "Plausibility:" in llm_response or "Testability:" in llm_response):
            logging.error("LLM response contains hypothesis report markers (Novelty/Plausibility/Testability)")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []

        # Improved pattern to match "Distinct Line of Thought" with optional number and colon
        # Matches: "Distinct Line of Thought 1:", "Distinct Line of Thought 1", "Distinct Line of Thought:", etc.
        # Use a more flexible pattern that handles variations (with or without numbers, with or without colons)
        pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"

        # First, try to find all occurrences of the pattern to verify we have 3 thoughts
        matches = list(re.finditer(pattern, llm_response, flags=re.IGNORECASE))

        logging.info(f"Found {len(matches)} 'Distinct Line of Thought' markers in response")

        if len(matches) >= 3:
            # We have at least 3 "Distinct Line of Thought" markers
            # Extract thoughts by finding text between markers
            thoughts = []
            for i, match in enumerate(matches):
                start_pos = match.end()  # Position after the marker
                # Find the end position (start of next marker, or end of string)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(llm_response)

                thought_text = llm_response[start_pos:end_pos].strip()
                if thought_text and len(thought_text) > 10:
                    thoughts.append(thought_text)

            # If we still have issues, try split as fallback
            if len(thoughts) < 3:
                raw_thoughts = re.split(pattern, llm_response, flags=re.IGNORECASE)
                thoughts = [t.strip() for t in raw_thoughts if t.strip() and len(t.strip()) > 10]

            # Remove the first element if it's just preamble (text before first "Distinct Line of Thought")
            if thoughts and len(thoughts) > 0:
                first_thought = thoughts[0]
                # If the first element doesn't look like a thought (too short),
                # and we have more thoughts, remove it
                if len(first_thought) < 30 and len(thoughts) > 1:
                    # Check if it looks like preamble (common words that appear before thoughts)
                    preamble_indicators = ['here', 'following', 'below', 'are', 'the', 'three', 'distinct', 'lines',
                                           'thoughts']
                    if any(indicator in first_thought.lower() for indicator in preamble_indicators) or (
                            first_thought and not first_thought[0].isupper()):
                        thoughts = thoughts[1:]

            # Ensure we have exactly 3 thoughts
            if len(thoughts) > 3:
                # Take the first 3 substantial thoughts
                substantial_thoughts = [t for t in thoughts if len(t.strip()) > 30]
                thoughts = substantial_thoughts[:3] if len(substantial_thoughts) >= 3 else thoughts[:3]
            elif len(thoughts) < 3 and len(thoughts) > 0:
                # If we have fewer than 3, check if any thought contains multiple "Distinct Line of Thought" markers
                expanded_thoughts = []
                for thought in thoughts:
                    # Check if this thought contains additional markers
                    sub_matches = list(re.finditer(pattern, thought, flags=re.IGNORECASE))
                    if len(sub_matches) > 0:
                        # Split this thought further
                        sub_split = re.split(pattern, thought, flags=re.IGNORECASE)
                        sub_split = [t.strip() for t in sub_split if t.strip() and len(t.strip()) > 10]
                        expanded_thoughts.extend(sub_split)
                    else:
                        expanded_thoughts.append(thought)

                if len(expanded_thoughts) >= 3:
                    thoughts = expanded_thoughts[:3]
        else:
            # Fallback: use simple split
            raw_thoughts = re.split(pattern, llm_response, flags=re.IGNORECASE)
            thoughts = [t.strip() for t in raw_thoughts if t.strip() and len(t.strip()) > 10]

            # Remove preamble if present
            if thoughts and len(thoughts) > 0:
                first_thought = thoughts[0]
                if len(first_thought) < 30 and len(thoughts) > 1:
                    preamble_indicators = ['here', 'following', 'below', 'are', 'the', 'three', 'distinct', 'lines',
                                           'thoughts']
                    if any(indicator in first_thought.lower() for indicator in preamble_indicators) or not \
                    first_thought[0].isupper():
                        thoughts = thoughts[1:]

            # Limit to 3
            if len(thoughts) > 3:
                substantial_thoughts = [t for t in thoughts if len(t.strip()) > 30]
                thoughts = substantial_thoughts[:3] if len(substantial_thoughts) >= 3 else thoughts[:3]

        # If no thoughts found, try alternative parsing
        if not thoughts or len(thoughts) == 0:
            # Try splitting by numbered lists (1., 2., 3.) or bullet points
            numbered_pattern = r'^\d+\.?\s+'
            lines = llm_response.splitlines()
            for line in lines:
                line = line.strip()
                if line and (re.match(numbered_pattern, line) or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets
                    clean_thought = re.sub(numbered_pattern, '', line)
                    clean_thought = re.sub(r'^[-•]\s*', '', clean_thought)
                    if clean_thought and len(clean_thought) > 20:  # Only substantial thoughts
                        thoughts.append(clean_thought)

            # If still no thoughts, try splitting by sentences
            if not thoughts:
                sentences = re.split(r'(?<=[.!?])\s+', llm_response)
                thoughts = [s.strip() for s in sentences if len(s.strip()) > 30][
                    :3]  # Take up to 3 substantial sentences

        # Ensure we have at least 3 thoughts (pad if needed)
        while len(thoughts) < 3:
            thoughts.append("")

        # --- Debug print ---
        logging.info(f"Parsed {len(thoughts)} thoughts from TOT response")
        for i, t in enumerate(thoughts[:3], 1):
            if t:  # Only print non-empty thoughts
                logging.info(f"Thought {i} (length: {len(t)}): {t[:100]}...")
                print(f"\n--- Thought {i} ---\n{t}\n")

        # Final validation: ensure we have exactly 3 thoughts, each substantial
        final_thoughts = []
        for t in thoughts[:3]:
            if t and t.strip() and len(t.strip()) > 20:
                final_thoughts.append(t.strip())
            else:
                final_thoughts.append("")  # Pad with empty if needed

        # Ensure exactly 3
        while len(final_thoughts) < 3:
            final_thoughts.append("")

        return _finalize_initial_thoughts(
            final_thoughts[:3], literature_context
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Use ASCII-safe logging to avoid UnicodeEncodeError on Windows console
        safe_error = str(e).encode('ascii', 'replace').decode('ascii')
        safe_details = error_details.encode('ascii', 'replace').decode('ascii')
        logging.error(f"TOT generation failed: {safe_error}")
        logging.error(f"Full traceback:\n{safe_details}")
        return [f"Error generating thoughts: {type(e).__name__}", "See console for details", ""]


def retry_thinking_deepen_thoughts(line_of_thought: str, previous_thought_1: str, previous_thought_2: str,
                                   initial_clarified_question: str, conversation_context: str = "",
                                   reasoning_policy: str = "auto"):
    # produces continuation options based on user's selected option
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        RETRY_THINKING_INSTRUCTIONS = instructions[4]

        # Build context section if available
        context_section = ""
        if conversation_context and conversation_context.strip():
            context_section = f"""

        **Conversation Context (for understanding the flow):**
        {conversation_context}

        **CRITICAL: Use this context to understand what has been discussed, but your 3 continuation options MUST build directly on the selected option below, not on earlier parts of the conversation.**
        """

        literature_context = get_local_literature_context(
            f"{initial_clarified_question}\nSelected continuation: {line_of_thought}",
            limit=4,
            require_direct_support=True,
            retrieval_attempts=2,
        )
        evidence_context = _merge_evidence_contexts(
            literature_context, initial_clarified_question
        )

        tot_generation_prompt = f"""
        {RETRY_THINKING_INSTRUCTIONS}

        **CRITICAL: The user has JUST SELECTED the following option/hypothesis. Your task is to generate EXACTLY 3 continuation options that explore this selected option further. These are NOT new starting points - they must continue from what the user selected.**

        **SELECTED OPTION (CONTINUE FROM THIS - DO NOT IGNORE THIS):**
        {line_of_thought}

        **Previous Options (NOT selected - shown for context only, do NOT continue from these):**
            Previous Option 1: {previous_thought_1}
            Previous Option 2: {previous_thought_2}

        **Initial Question (for context only):** {initial_clarified_question}
        {context_section}

        **LiteratureAgent evidence packet for this continuation:**
        {evidence_context}

        **FINAL REMINDER:**
        - The user selected the option above (the "SELECTED OPTION")
        - Generate 3 options that CONTINUE exploring that selected option
        - Each option should explore a different aspect of the SAME selected option
        - Do NOT generate new TOT thoughts or start new directions
        - Do NOT ignore the selected option and generate unrelated thoughts
        - Build directly on the materials, mechanisms, or conditions mentioned in the selected option
        - Novel entities and mechanisms are allowed; preserve them as proposals rather than filtering them out
        - If the selected option is unsupported, ask whether the proposed effect occurs and what would falsify it; do not convert the proposal into an established premise
        - Define an invented or ambiguous chemical identity before making structure-based claims about its functional groups, protonation, bonding, or packing
        - Ground a continuation only in a directly aligned [E#] claim that explicitly contains the same chemical entity, tested property, and reported relationship
        - [B#] and [KG#] items may provide context or analogy, but they cannot support the selected compound or mechanism
        - Any literature-grounded statement must include an `Evidence anchor:` with the aligned claim ID and DOI or paper_slug
        - Keep partial evidence within its tested scope; label unsupported extensions as `Novel proposal - not supported by the retrieved packet`
        - If the packet is unavailable, continue generating useful options but label them as unverified proposals
        - In standard layered lead-halide perovskites, a monovalent ammonium spacer is RP-type; do not label it DJ unless an explicitly defined divalent linker is present
        """

        if reasoning_policy and reasoning_policy.strip().lower() not in ("none", "off", "disabled"):
            tot_generation_prompt, _reasoning_meta = _apply_reasoning_policy(
                tot_generation_prompt,
                task=f"Deepen hypothesis exploration from: {line_of_thought[:200]}",
                task_type="hypothesis",
                policy=reasoning_policy,
            )
            _log_reasoning_metadata(_reasoning_meta, fn_name="retry_thinking_deepen_thoughts")

        # Log what we're sending to the LLM
        logging.info(f"retry_thinking_deepen_thoughts - Selected option (first 200 chars): {line_of_thought[:200]}")
        logging.info(f"retry_thinking_deepen_thoughts - Prompt length: {len(tot_generation_prompt)}")

        llm_response = generate_text_with_llm(tot_generation_prompt)

        if llm_response is None:
            llm_response = ""

        use_audit = _audit_and_store_literature_use(llm_response, evidence_context)
        if (
            _direct_evidence_count(evidence_context) > 0
            and use_audit["available_claim_ids"]
            and not use_audit["citation_complete"]
        ):
            repair_prompt = f"""
Revise the three continuation options below while preserving their direction and format.
Ground an option only in a directly aligned [E#] claim from the supplied LiteratureAgent
packet. Every literature-backed statement must include an `Evidence anchor:` naming that
[E#] claim and its DOI or paper_slug. Never use [B#] or [KG#] as validation.
Keep partial-context evidence within the variable actually tested. Clearly label all
unsupported extensions as `Novel proposal - not supported by the retrieved packet`.
Do not invent citations, results, materials, or numerical values.

LiteratureAgent packet:
{evidence_context}

Continuation options to revise:
{llm_response}
"""
            repaired = generate_text_with_llm(repair_prompt)
            if repaired and repaired.strip():
                llm_response = repaired
                _audit_and_store_literature_use(llm_response, evidence_context)

        # Log the raw response for debugging
        logging.info(f"retry_thinking_deepen_thoughts raw response (first 1000 chars): {llm_response[:1000]}")

        # Validate that the response actually references the selected option
        # Extract key terms from the selected option (improved to work with any chemical compound)
        import re as regex_module
        selected_lower = line_of_thought.lower()

        # Extract chemical compounds, molecular formulas, and key terms
        # Look for patterns like: Chemical names, formulas (e.g., CsPbI3, 2,2'-bipyridinium), cation markers (⁺, +)
        key_terms = []

        # Extract chemical abbreviations (e.g., XBDA, PPD, ATZ)
        abbrev_pattern = r'\b([A-Z]{2,6})\b'
        abbrevs = regex_module.findall(abbrev_pattern, line_of_thought)
        # Filter out action verbs and common words
        action_verbs = {'EXAMINING', 'EXPLORING', 'INVESTIGATING', 'QUANTIFYING', 'ANALYZING', 'COMPARE', 'EXAMINE'}
        abbrevs = [a for a in abbrevs if a.upper() not in action_verbs and 'CSPBI' not in a.upper()]
        key_terms.extend([a.lower() for a in abbrevs])

        # Extract words that look like chemical compounds (capitalized, contain numbers/symbols)
        chemical_pattern = r'\b[A-Z][a-z]*(?:\d+|[⁺₂₃₄])*(?:[A-Z][a-z]*(?:\d+|[⁺₂₃₄])*)*\b'
        chemicals = regex_module.findall(chemical_pattern, line_of_thought)
        # Filter out verbs
        chemicals = [c for c in chemicals if
                     c.lower() not in ['examining', 'exploring', 'investigating', 'quantifying', 'analyzing', 'compare',
                                       'examine', 'investigate']]
        key_terms.extend([c.lower() for c in chemicals if len(c) > 2])  # Filter short matches

        # Extract words with special characters that indicate chemical notation
        special_chem_pattern = r"[\w\d\-',]+(?:Cl₂|Br₂|I₂|⁺|₂|₃|₄)"
        special_chemicals = regex_module.findall(special_chem_pattern, line_of_thought)
        key_terms.extend([c.lower() for c in special_chemicals])

        # Extract hyphenated compound names (like 2,2'-bipyridinium, 1,4-bis...)
        hyphenated_pattern = r"\d+[,\-]\d+'?[,\-]?\w+"
        hyphenated = regex_module.findall(hyphenated_pattern, line_of_thought)
        key_terms.extend([c.lower() for c in hyphenated])

        # Remove duplicates, filter, and remove action verbs
        key_terms = list(set([term for term in key_terms if
                              len(term) > 3 and term not in ['examine', 'exploring', 'investigating', 'quantifying',
                                                             'analyzing', 'compare']]))

        logging.info(f"Extracted key terms from selected option: {key_terms[:10]}")  # Log first 10

        response_lower = llm_response.lower()
        if key_terms:
            # Check if ANY of the key terms appear in the response
            found_terms = [term for term in key_terms if term in response_lower]
            if len(found_terms) < max(1, len(key_terms) // 3):  # At least 1/3 of terms should be present
                logging.warning(f"WARNING: Generated response contains few/no key terms from selected option")
                logging.warning(f"Selected option mentions: {key_terms[:5]}, but response only found: {found_terms}")
                logging.warning(f"This suggests the LLM is generating unrelated thoughts instead of continuations!")
                logging.warning(f"Will use fallback to generate relevant continuations")

        # Parse response more robustly
        # Focus ONLY on extracting the 3 continuation options - IGNORE any socratic questions
        socratic_question = ""  # We don't want socratic questions for continuations
        options = []

        # Pattern to match "Distinct Line of Thought" format (same as tot_generation)
        thought_pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"

        # Find all "Distinct Line of Thought" markers - these are what we want
        thought_matches = list(re.finditer(thought_pattern, llm_response, flags=re.IGNORECASE))

        # IGNORE any socratic questions - we only want the 3 continuation options
        # Don't extract any questions, just skip to the options

        # Extract options using "Distinct Line of Thought" markers (same logic as tot_generation)
        if len(thought_matches) >= 3:
            # Extract thoughts by finding text between markers
            for i, match in enumerate(thought_matches):
                start_pos = match.end()  # Position after the marker
                # Find the end position (start of next marker, or end of string)
                if i + 1 < len(thought_matches):
                    end_pos = thought_matches[i + 1].start()
                else:
                    end_pos = len(llm_response)

                option_text = llm_response[start_pos:end_pos].strip()
                # Clean up: remove any remaining "Distinct Line of Thought" markers that might be embedded
                option_text = re.sub(thought_pattern, "", option_text, flags=re.IGNORECASE).strip()
                if option_text and len(option_text) > 10:
                    options.append(option_text)

            logging.info(f"Extracted {len(options)} options from 'Distinct Line of Thought' markers")
        elif len(thought_matches) > 0:
            # We have some markers but less than 3, try split approach
            raw_options = re.split(thought_pattern, llm_response, flags=re.IGNORECASE)
            options = [opt.strip() for opt in raw_options if opt.strip() and len(opt.strip()) > 10]
            # Remove preamble if present
            if options and len(options) > 0:
                first_opt = options[0]
                if len(first_opt) < 30 and len(options) > 1:
                    preamble_indicators = ['here', 'following', 'below', 'are', 'the', 'three', 'distinct', 'lines',
                                           'thoughts']
                    if any(indicator in first_opt.lower() for indicator in preamble_indicators):
                        options = options[1:]

        # Fallback: if we don't have 3 options yet, try alternative parsing methods
        if len(options) < 3:
            # Try to find options by looking for numbered items or bullet points
            numbered_pattern = r'^\d+\.\s+'
            bullet_pattern = r'^[-•]\s+'

            for line in lines:
                line_stripped = line.strip()
                # Look for numbered options (1., 2., 3.)
                if re.match(numbered_pattern, line_stripped) or re.match(bullet_pattern, line_stripped):
                    # Remove numbering/bullet
                    clean_line = re.sub(numbered_pattern, '', line_stripped)
                    clean_line = re.sub(bullet_pattern, '', clean_line)
                    # Remove "Distinct Line of Thought" if present
                    clean_line = re.sub(thought_pattern, '', clean_line, flags=re.IGNORECASE).strip()
                    if clean_line and len(clean_line) > 15 and clean_line not in options:
                        options.append(clean_line)
                        if len(options) >= 3:
                            break
                # Also check for lines that start with "Option" or contain substantial content
                elif len(line_stripped) > 30 and not any(
                        kw in line_stripped.lower() for kw in ['socratic', 'question', 'reasoning']):
                    # Remove "Distinct Line of Thought" if present
                    clean_line = re.sub(thought_pattern, '', line_stripped, flags=re.IGNORECASE).strip()
                    if clean_line and len(clean_line) > 15 and clean_line not in options:
                        options.append(clean_line)
                        if len(options) >= 3:
                            break

            logging.info(f"After fallback parsing, found {len(options)} options")

        # Log what we extracted
        logging.info(f"retry_thinking_deepen_thoughts: Extracted {len(options)} options")
        for i, opt in enumerate(options, 1):
            logging.info(f"  Option {i} (length: {len(opt)}): {opt[:100]}...")

        # CRITICAL: Validate that the extracted options actually reference the selected option
        # Use the same key_terms we extracted earlier
        validated_options = []
        for opt in options:
            opt_lower = str(opt).lower()
            # Check if this option references materials/terms from the selected option
            # Use key_terms we extracted earlier with improved pattern matching
            if not key_terms:
                # If no key terms were extracted, accept the option
                validated_options.append(opt)
            else:
                # Check if option references ANY of the key terms from the selected option
                found_in_opt = [term for term in key_terms if term in opt_lower]
                if found_in_opt or len(key_terms) == 0:
                    validated_options.append(opt)
                else:
                    logging.warning(f"Rejected option that doesn't reference selected materials: {opt[:100]}...")
                    logging.warning(
                        f"Selected option contains: {key_terms[:5]}, but this option doesn't reference them")

        # If we lost options due to validation, create proper continuations
        if len(validated_options) < 3:
            logging.warning(
                f"Only {len(validated_options)} options passed validation. Creating continuations based on selected option.")

            # Extract the PRIMARY subject from the selected option (the main material/concept being discussed)
            # This should be what the option is ABOUT (the chemical compound), not action verbs
            primary_subject = None

            # Pattern 1: Look for text in parentheses (often contains full chemical names like "XBDA" or "1,4-bis...")
            paren_pattern = r'\(([^)]{5,100})\)'  # 5-100 chars in parentheses
            parens = re.findall(paren_pattern, line_of_thought)
            if parens:
                # Filter out things that are clearly not chemical names
                chem_names = [p for p in parens if
                              not any(word in p.lower() for word in ['e.g.', 'i.e.', 'such as', 'including'])]
                if chem_names:
                    primary_subject = chem_names[0]
                    logging.info(f"Extracted primary subject from parentheses: {primary_subject}")

            # Pattern 2: Look for abbreviated chemical names before parentheses (e.g., "XBDA" in "XBDA (1,4-bis...)")
            if not primary_subject:
                abbrev_pattern = r'\b([A-Z]{2,6})\s*\('  # 2-6 capital letters before parenthesis
                abbrevs = re.findall(abbrev_pattern, line_of_thought)
                if abbrevs:
                    # Filter out common non-chemical abbreviations
                    chem_abbrevs = [a for a in abbrevs if a not in ['CsPbI', 'ATZ', 'PPD'] or len(a) >= 3]
                    if chem_abbrevs:
                        primary_subject = chem_abbrevs[0]
                        logging.info(f"Extracted primary subject from abbreviation: {primary_subject}")

            # Pattern 3: Look for specific compound mentions
            if not primary_subject:
                # Look for explicit compound names after "Introducing", "Utilizing", etc.
                introducing_pattern = r"(?:Introducing|Utilizing|Incorporating|Using|Employing)\s+([A-Za-z0-9\-,'\(\)]+\s*(?:acid|amine|ammonium|cation|spacer)?)\s+(?:as|is|could|to)"
                match = re.search(introducing_pattern, line_of_thought, re.IGNORECASE)
                if match:
                    subject_candidate = match.group(1).strip()
                    # Make sure it's not a verb phrase
                    if not any(verb in subject_candidate.lower() for verb in
                               ['investigating', 'exploring', 'examining', 'preventing', 'introducing', 'comparing',
                                'water ingress', 'phase']):
                        primary_subject = subject_candidate
                        logging.info(f"Extracted primary subject from introducing pattern: {primary_subject}")

            # Pattern 4: Look for chemical formulas with subscripts (but not CsPbI3)
            if not primary_subject or len(primary_subject) < 5:
                chem_formula_pattern = r'\b[A-Z][a-z]?(?:\d+|[⁺⁻₀₁₂₃₄₅₆₇₈₉])+(?:[A-Z][a-z]?(?:\d+|[⁺⁻₀₁₂₃₄₅₆₇₈₉])*)*\b'
                formulas = re.findall(chem_formula_pattern, line_of_thought)
                # Filter out CsPbI3 (that's the base material)
                formulas = [f for f in formulas if 'cspbi' not in f.lower()]
                if formulas:
                    primary_subject = formulas[0]
                    logging.info(f"Extracted primary subject from formula: {primary_subject}")

            # Fallback: Take first 60 chars after first comma or "of"
            if not primary_subject or len(primary_subject) < 5 or any(
                    verb in primary_subject.lower() for verb in ['investigating', 'exploring', 'examining']):
                # Try to get the noun phrase, not the verb phrase
                if ',' in line_of_thought:
                    parts = line_of_thought.split(',')
                    if len(parts) > 1:
                        primary_subject = parts[1][:60].strip()
                else:
                    primary_subject = "the proposed approach"
                logging.info(f"Using fallback primary subject: {primary_subject}")

            # Validate primary_subject doesn't contain action verbs or generic phrases
            action_verbs = ['investigating', 'exploring', 'examining', 'quantifying', 'analyzing', 'examine',
                            'investigate', 'explore', 'compare', 'preventing', 'forming', 'creating']
            bad_phrases = ['water ingress', 'phase transition', 'phase segregation', 'bandgap tuning', 'the effect',
                           'the influence', 'the impact']
            primary_lower = primary_subject.lower() if primary_subject else ""

            # If primary_subject contains action verbs, bad phrases, or is too generic, use a better fallback
            is_bad = (any(verb in primary_lower for verb in action_verbs) or
                      any(phrase in primary_lower for phrase in bad_phrases) or
                      len(primary_subject) > 150)

            if is_bad:
                logging.warning(
                    f"Primary subject contains action verbs/bad phrases or is too long: {primary_subject[:100]}")
                # Try to extract just the chemical compound name from the selected option
                # Look for patterns like "stearic acid", "2-aminoethylphosphonic acid", etc.
                # Pattern: word ending in "acid", "amine", "ammonium", etc.
                chem_name_pattern = r'\b([a-z0-9\-,\(\)]+(?:acid|amine|ammonium|phosphonic|carboxylic|pyridine|phenyl|thiol))\b'
                chem_names = re.findall(chem_name_pattern, line_of_thought, re.IGNORECASE)
                if chem_names:
                    primary_subject = chem_names[0]
                    logging.info(f"Fixed primary subject to chemical name: {primary_subject}")
                else:
                    # Look for capitalized abbreviations
                    compound_pattern = r'\b([A-Z]{2,6}(?:-[A-Z]{2,6})?)\b'
                    compounds = re.findall(compound_pattern, line_of_thought)
                    # Filter out CsPbI and common words
                    compounds = [c for c in compounds if 'cspbi' not in c.lower() and c not in ['CsPbI', 'Pb', 'I']]
                    if compounds:
                        primary_subject = compounds[0]
                        logging.info(f"Fixed primary subject to abbreviation: {primary_subject}")
                    else:
                        primary_subject = "the proposed material"
                        logging.info(f"Using generic fallback: {primary_subject}")

            # Create SOCRATIC-STYLE continuation options (how/why/what questions)
            continuation_options = [
                f"How does {primary_subject} specifically interact at the molecular level with CsPbI₃, and what structural features enable these interactions?",
                f"Why is {primary_subject} expected to be more effective than alternative approaches, and what are the thermodynamic or kinetic advantages?",
                f"What are the potential challenges or limitations of using {primary_subject}, and how might these be addressed?"
            ]
            logging.info(f"Created socratic-style continuation options with subject: {primary_subject}")

            # Use validated options first, then fill with proper continuations
            options = validated_options + continuation_options[:3 - len(validated_options)]

        # Ensure we have exactly 3 options
        if len(options) < 3:
            remaining = 3 - len(options)
            generic_continuations = [
                "Analyze the electronic structure modifications that would result from this approach",
                "Explore experimental conditions and synthesis parameters for optimal phase control",
                "Investigate the relationship between structural ordering and photoluminescence narrowing"
            ]
            options.extend(generic_continuations[:remaining])
        elif len(options) > 3:
            options = options[:3]  # Take only first 3

        # Ensure all options are non-empty and substantial, and don't contain "Distinct Line of Thought" markers
        final_options = []
        for i, opt in enumerate(options):
            if opt and opt.strip() and len(opt.strip()) >= 5:
                # Clean up: remove any "Distinct Line of Thought" markers that might be at the start
                clean_opt = re.sub(thought_pattern, "", opt, flags=re.IGNORECASE).strip()
                if clean_opt and len(clean_opt) >= 5:
                    final_options.append(clean_opt)

        # Ensure we have exactly 3
        while len(final_options) < 3:
            final_options.append(f"Option {len(final_options) + 1}: Continue exploring")

        options = final_options[:3]

        # Final validation
        for i in range(len(options)):
            if not options[i] or len(options[i].strip()) < 5:
                options[i] = f"Option {i + 1}: Continue exploring this line of reasoning"

        options = _gate_continuation_options(options, line_of_thought, evidence_context)

        # Log for debugging
        logging.info(
            f"retry_thinking_deepen_thoughts returning: {len(options)} continuation options (no socratic question for continuations)")

        # Always return empty socratic question for continuations - we don't want new questions
        return "", options
    except Exception as e:
        logging.error(f"Retry Thinking generation failed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

        # Create meaningful fallback options based on the selected hypothesis
        selected_text = line_of_thought[:200] if line_of_thought else ""

        # Try to extract key materials/mechanisms mentioned
        material_patterns = [r'(\w+[⁺²⁺]?)', r'([A-Z][a-z]+\w*[⁺²⁺]?)']
        materials = []
        for pattern in material_patterns:
            matches = re.findall(pattern, selected_text)
            materials.extend([m for m in matches if len(m) > 2 and m not in materials][:2])

        if materials:
            mat1 = materials[0] if len(materials) > 0 else "the proposed material"
            fallback_options = [
                f"Explore how {mat1} concentration and ratio affect the phase distribution and bandgap tuning",
                f"Investigate the specific mechanism by which {mat1} interacts with the inorganic framework",
                f"Compare {mat1} with alternative co-spacers to determine optimal properties for the target bandgap"
            ]
        else:
            fallback_options = [
                "Explore how the concentration and ratio of the proposed co-spacer affect phase distribution and bandgap tuning",
                "Investigate the specific mechanism of interaction with the inorganic framework to promote narrowed PL emission",
                "Compare with alternative co-spacers to determine optimal structural and electronic properties for the target bandgap"
            ]

        # Return empty socratic question and fallback options
        return "", fallback_options


def hypothesis_synthesis(socratic_question: str, next_step_option: str, previous_option_1: str,
                         previous_option_2: str, conversation_context: str, reasoning_policy: str = "auto",
                         literature_context_override: str | None = None) -> str:
    # Produce hypothesis based on user selected thoughts and other information
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        HYPOTHESIS_SYNTHESIS = instructions[5]

        literature_context = str(literature_context_override or "").strip()
        if literature_context:
            _mark_synthesis_evidence_context(literature_context)
        else:
            literature_context = _reuse_selected_branch_context(next_step_option)
        if not literature_context:
            literature_context = get_local_literature_context(
                f"{socratic_question}\nSelected continuation: {next_step_option}",
                limit=5,
                require_direct_support=True,
                retrieval_attempts=2,
            )
        _mark_synthesis_evidence_context(literature_context)
        evidence_context = _merge_evidence_contexts(
            literature_context, socratic_question, conversation_context
        )
        hypothesis_synthesis_prompt = f"""
            {HYPOTHESIS_SYNTHESIS}
            Socratic Question: {socratic_question}
            Next-Step Option: {next_step_option}
            Previous Step Options:
                Previous Option 1: {previous_option_1}
                Previous Option 2: {previous_option_2}
            Full Conversation Context: {conversation_context}
            Evidence-grounded local literature packet:
            {evidence_context}

            Preserve creative, novel proposals while separating them from evidence-supported claims.
            Ground directly supported statements in supplied evidence and preserve provenance identifiers.
            For a newly proposed chemical entity, state the intended identity or structural assumptions
            before inferring behavior from functional groups, protonation, bonding, or packing. Label every
            unsupported extrapolation as a proposed mechanism. An evidence anchor is relevant only when its
            reported claim bears on the same property or mechanism; do not use an unrelated citation merely
            because it concerns perovskites.
            """

        _wrapped = _generate_with_reasoning_wrapper(
            base_prompt=hypothesis_synthesis_prompt,
            task=f"Synthesize hypothesis from: {next_step_option[:300]}",
            task_type="hypothesis",
            policy=reasoning_policy,
            fn_name="hypothesis_synthesis",
        )
        llm_response = _wrapped["response"]

        if llm_response is None or not llm_response.strip():
            return "Error generating hypothesis. Please check your API key and try again."

        use_audit = _audit_and_store_literature_use(llm_response, evidence_context)
        if (
            _direct_evidence_count(evidence_context) > 0
            and use_audit["available_claim_ids"]
            and not use_audit["citation_complete"]
        ):
            repair_prompt = f"""
Revise the hypothesis below without changing its core scientific proposal. Only a directly
aligned [E#] claim may support the selected compound and mechanism. Any statement presented
as literature-supported must include that [E#] claim and its DOI or paper_slug. Never use
[B#] or [KG#] as validation. Preserve the tested scope of partial
            evidence and add an `Evidence boundary:` where the source does not test the proposed
            variable directly. Keep novel ideas, but label them `Novel proposal - not supported by
            the retrieved packet`. Novel entities may be invented, but their identity and structural
            assumptions must be explicit and must not be attributed to a source that did not study them.
            Remove evidence anchors that concern a different property or mechanism. Do not invent
            citations, reported facts, or unsupported numerical effect sizes.

LiteratureAgent packet:
{evidence_context}

Hypothesis to revise:
{llm_response}
"""
            repaired = generate_text_with_llm(repair_prompt)
            if repaired and repaired.strip():
                llm_response = repaired
                _audit_and_store_literature_use(llm_response, evidence_context)

        return llm_response

    except Exception as e:
        logging.error(f"Hypothesis generation failed: {e}")
        return f"Error generating hypothesis: {str(e)}. Please check your API key and try again."


def analyze_hypothesis(hypothesis: str, socratic_question: str, rubric_weights=None):
    """Score a hypothesis against local LiteratureAgent evidence and optional external retrieval."""
    if rubric_weights is None:
        rubric_weights = {"novelty": 1 / 3, "plausibility": 1 / 3, "testability": 1 / 3}

    local_evidence = get_local_literature_context(f"{socratic_question}\n{hypothesis}", limit=5)
    external_evidence = ""
    if FUTUREHOUSE_API_KEY:
        try:
            EdisonClient, JobNames, TaskRequest, RuntimeConfig = _lazy_import_edison()
            client = EdisonClient(api_key=FUTUREHOUSE_API_KEY)
            task_data = TaskRequest(
                name=JobNames.CROW,
                query=socratic_question,
                runtime_config=RuntimeConfig(max_steps=5)
            )
            responses = client.run_tasks_until_done(task_data)
            external_evidence = responses[0].formatted_answer.strip()
        except Exception as exc:
            logging.warning(f"External literature retrieval unavailable; using LiteratureAgent only: {exc}")
    evidence_summary = (
        f"LOCAL LITERATUREAGENT EVIDENCE:\n{local_evidence}\n\n"
        f"EXTERNAL RETRIEVAL EVIDENCE:\n{external_evidence or 'not available'}"
    )

    prior_matches = len(re.findall(re.escape(hypothesis), evidence_summary, flags=re.IGNORECASE))
    novelty_score = max(0.0, 1.0 - min(prior_matches, 10) * 0.1)
    novelty_reason = (
        "High novelty: little to no direct matches found in literature."
        if novelty_score > 0.8 else
        "Moderate novelty: related ideas appear in literature."
        if 0.4 < novelty_score <= 0.8 else
        "Low novelty: hypothesis closely resembles existing findings."
    )

    key_terms = re.findall(r"\b\w+\b", hypothesis)
    hits = sum(1 for term in key_terms if re.search(rf"\b{re.escape(term)}\b", evidence_summary, flags=re.IGNORECASE))
    plausibility_score = min(1.0, hits / max(len(key_terms), 1))
    plausibility_reason = (
        "Strong conceptual support in related studies."
        if plausibility_score > 0.7 else
        "Partial conceptual alignment with existing evidence."
        if 0.4 < plausibility_score <= 0.7 else
        "Weak or minimal evidence supporting the hypothesis."
    )

    test_keywords = ["experiment", "measure", "observe", "compare", "test", "quantify", "analyze"]
    found_keywords = [kw for kw in test_keywords if re.search(rf"\b{kw}\b", evidence_summary, re.IGNORECASE)]
    testability_score = 1.0 if found_keywords else 0.5
    testability_reason = (
        "Evidence discusses measurable or experimental approaches."
        if found_keywords else
        "Limited direct mention of testable or measurable outcomes."
    )

    scores = {
        "novelty": novelty_score,
        "plausibility": plausibility_score,
        "testability": testability_score
    }
    weighted_score = sum(scores[k] * rubric_weights[k] for k in rubric_weights)

    report = f"""
### Hypothesis Evaluation Report

**Hypothesis:**
{hypothesis}

---

#### Evidence Summary:
{evidence_summary}

---

#### Scoring Breakdown:

| Criterion | Score | Explanation |
|:-----------|:------:|:------------|
| **Novelty** | {novelty_score:.2f} | {novelty_reason} |
| **Plausibility** | {plausibility_score:.2f} | {plausibility_reason} |
| **Testability** | {testability_score:.2f} | {testability_reason} |

**Weighted Total Score:** {weighted_score:.2f}

---

#### Interpretation:
- **> 0.80:** Highly original, plausible, and testable.
- **0.50 – 0.80:** Promising but may need refinement or stronger evidence.
- **< 0.50:** Likely weak, redundant, or insufficiently grounded.

---

*Generated from LiteratureAgent local evidence, optional external retrieval,
and heuristic rubric evaluation.*
"""

    return {
        "evidence_summary": evidence_summary,
        "scores": scores,
        "weighted_score": weighted_score,
        "report": report.strip()
    }


def local_hypothesis_analysis_fallback(
    hypothesis: str,
    socratic_question: str,
    literature_context_override: str | None = None,
) -> str:
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        HYPOTHESIS_ANALYSIS_REPORT = instructions[6]

        handoff = get_last_literature_handoff() or {}
        literature_context = str(literature_context_override or "").strip()
        if handoff.get("used_for_hypothesis_synthesis"):
            literature_context = literature_context or str(handoff.get("formatted_context") or "").strip()
            if literature_context:
                handoff["analysis_reused_synthesis_packet"] = True
                _last_literature_handoff.set(handoff)
        if not literature_context:
            literature_context = get_local_literature_context(
                f"{socratic_question}\n{hypothesis}",
                limit=5,
            )
        hypothesis_analysis_report_prompt = f"""
        {HYPOTHESIS_ANALYSIS_REPORT}
        Hypothesis: {hypothesis}
        Socratic Question: {socratic_question}

        Evidence-grounded local literature context:
        {literature_context}

        Evaluate the hypothesis without suppressing creative novelty. Distinguish direct evidence,
        analogy, inference, and novel proposal, and retain DOI/paper identifiers when available.
        Report novelty only relative to the retrieved packet unless a dedicated prior-art search supports
        a broader claim. If a proposed chemical identity is new or ambiguous, require its intended structure,
        formula, protonation state, and counterion before evaluating structure-based mechanisms.
        Treat an evidence anchor as support only when the reported claim tests the same property or mechanism;
        otherwise identify it as irrelevant or contextual rather than using it to raise plausibility.
        Do not infer stronger bonding from reduced XRD spacing alone, defect density from AFM morphology alone,
        or reduced nonradiative recombination from a longer PL lifetime alone. Do not praise an experimental
        design as robust without controls and discriminating measurements. Remove arbitrary percentage targets
        unless they are justified by prior data, measurement precision, or statistical power. Conclude with a
        calibrated status such as evidence-supported, indirectly motivated, or speculative but testable.
        """

        _wrapped = _generate_with_reasoning_wrapper(
            base_prompt=hypothesis_analysis_report_prompt,
            task=f"Analyze hypothesis against literature evidence: {hypothesis[:200]}",
            task_type="hypothesis",
            policy="auto",
            fn_name="local_hypothesis_analysis_fallback",
        )
        llm_response = _wrapped["response"]

        return llm_response

    except Exception as e:
        logging.error(f"Hypothesis Analysis Report generation failed: {e}")
        return f"Error generating hypothesis analysis: {str(e)}. Please check your API key and try again."


# POLARIS_HYPOTHESIS_OUTPUT_GUARD_V1
from functools import wraps as _polaris_wraps

try:
    from app.tools.hypothesis_quality import guard_hypothesis_output as _polaris_guard_hypothesis_output
except ImportError:
    from hypothesis_quality import guard_hypothesis_output as _polaris_guard_hypothesis_output

_polaris_original_hypothesis_synthesis = hypothesis_synthesis


@_polaris_wraps(_polaris_original_hypothesis_synthesis)
def hypothesis_synthesis(*args, **kwargs):
    return _polaris_guard_hypothesis_output(
        _polaris_original_hypothesis_synthesis(*args, **kwargs)
    )


from app.tools.hypothesis_scientific_guard import (
    guard_hypothesis_analysis as _polaris_guard_hypothesis_analysis,
    guard_hypothesis_synthesis as _polaris_guard_hypothesis_synthesis,
    wrap_clean_model_output as _polaris_wrap_clean_model_output,
)

_polaris_scientific_original_synthesis = hypothesis_synthesis


@_polaris_wraps(_polaris_scientific_original_synthesis)
def hypothesis_synthesis(*args, **kwargs):
    return _polaris_guard_hypothesis_synthesis(
        _polaris_scientific_original_synthesis(*args, **kwargs)
    )


def _polaris_analysis_wrapper(function):
    @_polaris_wraps(function)
    def wrapped(*args, **kwargs):
        context = "\n".join(
            [*(str(value) for value in args), *(str(value) for value in kwargs.values())]
        )
        return _polaris_guard_hypothesis_analysis(function(*args, **kwargs), context)

    return wrapped


for _polaris_analysis_name in (
    "hypothesis_analysis_report",
    "generate_hypothesis_analysis_report",
    "analyze_hypothesis",
):
    if callable(globals().get(_polaris_analysis_name)):
        globals()[_polaris_analysis_name] = _polaris_analysis_wrapper(
            globals()[_polaris_analysis_name]
        )

for _polaris_tot_name in (
    "socratic_pass_and_tot",
    "socratic_question_and_tot",
    "tot_generation",
):
    if callable(globals().get(_polaris_tot_name)):
        globals()[_polaris_tot_name] = _polaris_wrap_clean_model_output(
            globals()[_polaris_tot_name]
        )

# Final-report live quality control is intentionally bounded to one report edit.
try:
    import functools as _quality_functools
    import inspect as _quality_inspect
    import logging as _quality_logging

    from app.services.hypothesis_quality_control import (
        run_hypothesis_quality_cycle as _run_hypothesis_quality_cycle,
    )

    _quality_logger = _quality_logging.getLogger(__name__)
    _analyze_hypothesis_before_live_quality = globals().get("analyze_hypothesis")

    def _quality_model_adapter(policy_name, memory):
        def _call(prompt):
            helper = globals().get("_generate_with_reasoning_wrapper")
            if not callable(helper):
                raise RuntimeError("Reasoning-aware generation helper is unavailable")

            signature = _quality_inspect.signature(helper)
            values = {}
            unsupported = []
            for name, parameter in signature.parameters.items():
                lowered = name.lower()
                if parameter.kind in (
                    _quality_inspect.Parameter.VAR_POSITIONAL,
                    _quality_inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if "prompt" in lowered or lowered in {"message", "input", "text"}:
                    values[name] = prompt
                elif "policy" in lowered or "philosoph" in lowered:
                    values[name] = policy_name
                elif "task" in lowered or lowered in {"stage", "agent_name"}:
                    values[name] = "hypothesis_quality_control"
                elif "memory" in lowered or lowered in {"state", "manager"}:
                    values[name] = memory
                elif lowered in {"function", "func", "generator", "base_function"}:
                    values[name] = _analyze_hypothesis_before_live_quality
                elif parameter.default is _quality_inspect.Parameter.empty:
                    unsupported.append(name)
            if unsupported:
                raise TypeError(
                    "Unsupported reasoning helper parameters: " + ", ".join(unsupported)
                )

            result = helper(**values)
            if callable(result):
                result = result(prompt)
            if isinstance(result, dict):
                result = (
                    result.get("response")
                    or result.get("content")
                    or result.get("text")
                    or result.get("output")
                )
            if not isinstance(result, str) or not result.strip():
                raise RuntimeError("Reasoning-aware generation returned no text")
            return result

        return _call

    if callable(_analyze_hypothesis_before_live_quality):

        @_quality_functools.wraps(_analyze_hypothesis_before_live_quality)
        def analyze_hypothesis(*args, **kwargs):
            report = _analyze_hypothesis_before_live_quality(*args, **kwargs)
            if not isinstance(report, str) or report.lstrip().lower().startswith("error"):
                return report

            context_parts = [str(value) for value in args if value is not None]
            context_parts.extend(
                f"{key}: {value}" for key, value in kwargs.items() if value is not None
            )
            hypothesis_context = "\n\n".join(context_parts)
            memory = None
            try:
                from app.services.memory_service import get_memory_manager

                memory = get_memory_manager()
            except Exception:
                _quality_logger.debug(
                    "Hypothesis quality control could not access shared memory",
                    exc_info=True,
                )

            try:
                enabled = True
                if memory is not None:
                    enabled = bool(
                        memory.get_var("hypothesis_live_auditor_enabled", True)
                    )
                if not enabled:
                    return report
                final_report, _ = _run_hypothesis_quality_cycle(
                    report=report,
                    hypothesis_context=hypothesis_context,
                    auditor_llm=_quality_model_adapter("humean_evidence", memory),
                    editor_llm=_quality_model_adapter("kantian_constraints", memory),
                    memory=memory,
                )
                return final_report
            except Exception:
                _quality_logger.warning(
                    "Hypothesis live quality control failed; returning the guarded report",
                    exc_info=True,
                )
                return report
except Exception:
    logging.getLogger(__name__).warning(
        "Hypothesis live quality control registration failed", exc_info=True
    )


# The streaming Hypothesis workflow uses the local analysis fallback directly.
try:
    import functools as _live_quality_functools
    import logging as _live_quality_logging

    from app.services.hypothesis_quality_control import (
        run_hypothesis_quality_cycle as _run_live_hypothesis_quality_cycle,
    )

    _live_quality_logger = _live_quality_logging.getLogger(__name__)
    _local_analysis_before_live_quality = globals().get(
        "local_hypothesis_analysis_fallback"
    )

    if callable(_local_analysis_before_live_quality):

        @_live_quality_functools.wraps(_local_analysis_before_live_quality)
        def local_hypothesis_analysis_fallback(*args, **kwargs):
            report = _local_analysis_before_live_quality(*args, **kwargs)
            if not isinstance(report, str) or report.lstrip().lower().startswith("error"):
                return report

            context_parts = [str(value) for value in args if value is not None]
            context_parts.extend(
                f"{key}: {value}" for key, value in kwargs.items() if value is not None
            )
            memory = None
            try:
                from app.services.memory_service import get_memory_manager

                memory = get_memory_manager()
                view_component = getattr(memory, "view_component", None)
                if callable(view_component):
                    for component in (
                        "initial_question",
                        "clarified_question",
                        "last_selected_option",
                    ):
                        value = view_component(component)
                        if value:
                            context_parts.append(f"{component}: {value}")
            except Exception:
                _live_quality_logger.debug(
                    "Live Hypothesis quality control could not access shared memory",
                    exc_info=True,
                )

            try:
                enabled = True
                if memory is not None:
                    enabled = bool(
                        memory.get_var("hypothesis_live_auditor_enabled", True)
                    )
                if not enabled:
                    return report
                adapter = globals().get("_quality_model_adapter")
                auditor_llm = adapter("humean_evidence", memory) if callable(adapter) else None
                editor_llm = adapter("kantian_constraints", memory) if callable(adapter) else None
                final_report, _ = _run_live_hypothesis_quality_cycle(
                    report=report,
                    hypothesis_context="\n\n".join(context_parts),
                    auditor_llm=auditor_llm,
                    editor_llm=editor_llm,
                    memory=memory,
                )
                return final_report
            except Exception:
                _live_quality_logger.warning(
                    "Live Hypothesis quality control failed; returning the guarded report",
                    exc_info=True,
                )
                return report
except Exception:
    logging.getLogger(__name__).warning(
        "Live Hypothesis quality-control registration failed", exc_info=True
    )


from app.services.hypothesis_branch_grounding import (
    install_hypothesis_grounding as _install_hypothesis_grounding,
)

_install_hypothesis_grounding(globals())
del _install_hypothesis_grounding
