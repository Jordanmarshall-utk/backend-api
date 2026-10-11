from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "good",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "with",
}


@dataclass(frozen=True)
class LiteratureKGClaim:
    claim_id: str
    claim_text: str
    predicate: str
    causal_status: str
    support_status: str
    doi: str
    paper_title: str
    subjects: tuple[str, ...]
    objects: tuple[str, ...]
    evidence_excerpt: str
    score: float


@dataclass
class _GraphIndex:
    snapshot_id: str
    claims: dict[str, dict]
    papers: dict[str, dict]
    evidence: dict[str, dict]
    entities: dict[str, dict]
    claim_papers: dict[str, str]
    claim_evidence: dict[str, list[str]]
    claim_subjects: dict[str, list[str]]
    claim_objects: dict[str, list[str]]
    documents: dict[str, frozenset[str]]


def _default_graph_root() -> Path:
    configured = os.getenv("LITERATURE_KG_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[4]
    return (
        project_root
        / "artifacts"
        / "literature_agent_pce_stability"
        / "source_data"
        / "knowledge_graph"
        / "full_corpus_standardized_v1"
    )


def _tokens(value: str) -> frozenset[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(value.casefold()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        tokens.add(token)
        if len(token) > 4 and token.endswith("s"):
            tokens.add(token[:-1])
    return frozenset(tokens)


def _properties(row: dict[str, str]) -> dict:
    raw = row.get("properties_json", "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=4)
def _load_index(root_value: str) -> _GraphIndex:
    root = Path(root_value)
    nodes_path = root / "neo4j" / "nodes.csv"
    relationships_path = root / "neo4j" / "relationships.csv"
    if not nodes_path.is_file() or not relationships_path.is_file():
        raise FileNotFoundError(f"Literature knowledge graph not found at {root}")

    claims: dict[str, dict] = {}
    papers: dict[str, dict] = {}
    evidence: dict[str, dict] = {}
    entities: dict[str, dict] = {}
    snapshot_id = "unknown"

    with nodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            node_id = row.get("id:ID", "")
            label = row.get(":LABEL", "")
            props = _properties(row)
            if label == "ScientificClaim":
                claims[node_id] = props
            elif label == "Paper":
                papers[node_id] = props
            elif label == "Evidence":
                evidence[node_id] = props
            elif label == "ScientificEntity":
                entities[node_id] = props
            elif label == "DatasetSnapshot":
                snapshot_id = str(props.get("snapshot_id", snapshot_id))

    claim_papers: dict[str, str] = {}
    claim_evidence: dict[str, list[str]] = {}
    claim_subjects: dict[str, list[str]] = {}
    claim_objects: dict[str, list[str]] = {}

    with relationships_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            start = row.get(":START_ID", "")
            end = row.get(":END_ID", "")
            relation = row.get(":TYPE", "")
            if relation == "REPORTS_CLAIM":
                claim_papers[end] = start
            elif relation == "SUPPORTED_BY" and start in claims:
                claim_evidence.setdefault(start, []).append(end)
            elif relation == "HAS_SUBJECT":
                claim_subjects.setdefault(start, []).append(end)
            elif relation == "HAS_OBJECT":
                claim_objects.setdefault(start, []).append(end)

    documents: dict[str, frozenset[str]] = {}
    for claim_id, claim in claims.items():
        paper = papers.get(claim_papers.get(claim_id, ""), {})
        evidence_text = " ".join(
            str(evidence.get(item, {}).get("evidence_text", ""))
            for item in claim_evidence.get(claim_id, ())
        )
        entity_text = " ".join(
            str(entities.get(item, {}).get("label", ""))
            for item in (
                claim_subjects.get(claim_id, ()) + claim_objects.get(claim_id, ())
            )
        )
        documents[claim_id] = _tokens(
            " ".join(
                (
                    str(claim.get("claim_text", "")),
                    str(claim.get("predicate", "")),
                    str(paper.get("title", "")),
                    entity_text,
                    evidence_text,
                )
            )
        )

    return _GraphIndex(
        snapshot_id=snapshot_id,
        claims=claims,
        papers=papers,
        evidence=evidence,
        entities=entities,
        claim_papers=claim_papers,
        claim_evidence=claim_evidence,
        claim_subjects=claim_subjects,
        claim_objects=claim_objects,
        documents=documents,
    )


def _entity_labels(index: _GraphIndex, node_ids: list[str]) -> tuple[str, ...]:
    labels = {
        str(index.entities.get(node_id, {}).get("label", "")).strip()
        for node_id in node_ids
    }
    return tuple(sorted(label for label in labels if label))


def retrieve_literature_kg_claims(
    query: str,
    limit: int = 6,
    graph_root: str | Path | None = None,
) -> list[LiteratureKGClaim]:
    query_tokens = _tokens(query)
    if not query_tokens or limit < 1:
        return []
    root = Path(graph_root) if graph_root is not None else _default_graph_root()
    index = _load_index(str(root.resolve()))
    ranked: list[tuple[float, str]] = []
    for claim_id, document_tokens in index.documents.items():
        overlap = query_tokens & document_tokens
        if not overlap:
            continue
        coverage = len(overlap) / len(query_tokens)
        specificity = sum(1.0 + min(len(token), 12) / 12 for token in overlap)
        ranked.append((specificity + coverage * 4, claim_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    results: list[LiteratureKGClaim] = []
    seen_claims: set[tuple[str, str]] = set()
    for score, claim_id in ranked:
        claim = index.claims[claim_id]
        paper = index.papers.get(index.claim_papers.get(claim_id, ""), {})
        claim_text = str(claim.get("claim_text", "")).strip()
        doi = str(claim.get("doi") or paper.get("doi", "")).strip()
        identity = (doi.casefold(), " ".join(claim_text.casefold().split()))
        if identity in seen_claims:
            continue
        seen_claims.add(identity)
        evidence_parts = [
            str(index.evidence.get(item, {}).get("evidence_text", "")).strip()
            for item in index.claim_evidence.get(claim_id, ())
        ]
        evidence_excerpt = " ".join(part for part in evidence_parts if part)
        if len(evidence_excerpt) > 420:
            evidence_excerpt = evidence_excerpt[:417].rstrip() + "..."
        results.append(
            LiteratureKGClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                predicate=str(claim.get("predicate", "reported_relation")).strip(),
                causal_status=str(claim.get("causal_status", "not_causal")).strip(),
                support_status=str(claim.get("support_status", "unsupported")).strip(),
                doi=doi,
                paper_title=str(paper.get("title", "")).strip(),
                subjects=_entity_labels(
                    index, index.claim_subjects.get(claim_id, [])
                ),
                objects=_entity_labels(index, index.claim_objects.get(claim_id, [])),
                evidence_excerpt=evidence_excerpt,
                score=round(score, 4),
            )
        )
        if len(results) >= limit:
            break
    return results


def build_hypothesis_kg_context(
    query: str,
    limit: int = 6,
    graph_root: str | Path | None = None,
) -> str:
    try:
        claims = retrieve_literature_kg_claims(query, limit, graph_root)
        root = Path(graph_root) if graph_root is not None else _default_graph_root()
        index = _load_index(str(root.resolve()))
    except (FileNotFoundError, OSError, csv.Error, json.JSONDecodeError):
        return ""
    if not claims:
        return ""
    lines = [
        f"LITERATURE_KG_CONTEXT snapshot={index.snapshot_id}",
        "These are paper-reported claims or associations, not causal proof.",
    ]
    for position, claim in enumerate(claims, 1):
        entities = " -> ".join((*claim.subjects, *claim.objects)) or "not resolved"
        lines.extend(
            (
                f"[KG{position}] {claim.claim_text}",
                f"DOI: {claim.doi or 'not available'}",
                f"Relation: {claim.predicate}; causal_status={claim.causal_status}; "
                f"support_status={claim.support_status}",
                f"Entities: {entities}",
                f"Evidence excerpt: {claim.evidence_excerpt or 'not retained'}",
            )
        )
    return "\n".join(lines)
