"""POLARIS-facing service contract for LiteratureAgent.

The service deliberately uses files and subprocesses rather than importing the
large embedded LiteratureAgent runtime into the long-lived Streamlit process.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from app.tools.agent_contract import AgentResult, AgentTask
except ImportError:
    from agent_contract import AgentResult, AgentTask


STAGES = {"extract_batch", "vision_pass", "sanitize_summaries", "integrate_and_model", "knowledge_graph", "evidence_packet"}
EXTRACTION_PROFILES = {"none", "hypothesis_support", "general_literature", "pce_stability_modeling"}
DOMAIN_PROFILES = {"general_science", "oxide_perovskite", "halide_pv"}
HALIDE_PV_DOMAIN_PROFILE = "halide_pv"
PCE_EXTRACTION_PROFILE = "pce_stability_modeling"
NON_PV_FAMILY_GATING = "off"

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_STACK_ROOT = _BACKEND_ROOT.parent
_LITERATURE_ROOT = _BACKEND_ROOT / "literature_agent"
_RUNTIME_ROOT = _STACK_ROOT / "runtime" / "literature_agent"

EVIDENCE_QUERY_STOPWORDS = {
    "about", "according", "also", "analysis", "and", "answer", "are", "based", "can",
    "been", "being", "between", "could", "direct", "does", "evidence", "experiment",
    "experiments", "explain", "for", "formation", "from", "further", "have", "having",
    "continue", "exploring", "how", "hypothesis", "hypothesized", "idea", "identify", "interpretation", "into", "literature",
    "mechanism", "mechanistic", "needed", "paper", "papers", "possible", "promote",
    "question", "report", "reported", "research", "result", "results", "say", "should", "systematic",
    "show", "studies", "study", "support", "test", "testing", "that", "the", "their", "type",
    "them", "there", "these", "this", "through", "under", "use", "used", "using",
    "underlying", "what", "when", "where", "whether", "which", "while", "with", "would",
}

QUERY_GENERIC_TERMS = {
    "approach", "change", "changes", "effect", "effects", "impact", "improve", "improved",
    "material", "materials", "property", "properties", "quality", "system", "systems", "variation",
}

QUERY_GENERIC_CONCEPTS = {
    "characterization", "device", "experimental", "performance", "processing", "stability", "structural",
}

QUERY_GENERIC_MATCHES = {
    "alpha phase", "black phase", "device", "film", "films", "material", "materials", "performance",
    "perovskite", "perovskites", "phase", "processing", "stability",
}


def normalize_domain_profile(value: Any) -> str:
    """Return a supported LiteratureAgent domain-profile identifier."""
    profile = str(value or "").strip()
    if profile not in DOMAIN_PROFILES:
        raise ValueError(
            f"Unsupported domain_profile: {profile or '[empty]'}. "
            f"Expected one of: {sorted(DOMAIN_PROFILES)}"
        )
    return profile


def _resolve_request_ontology_path(value: Any) -> str:
    """Resolve an API-supplied ontology without permitting backend file escape."""
    raw_path = str(value or "").strip()
    if not raw_path:
        raise ValueError("ontology_path must be a non-empty backend-local JSON path")

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = _BACKEND_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "ontology_path must reference an existing, readable backend-local JSON file"
        ) from exc

    backend_root = _BACKEND_ROOT.resolve()
    if not resolved.is_relative_to(backend_root):
        raise ValueError("ontology_path must stay within the POLARIS backend directory")
    if resolved.suffix.lower() != ".json" or not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise ValueError(
            "ontology_path must reference an existing, readable backend-local JSON file"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("ontology_path must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("ontology_path must contain one top-level JSON object")
    return str(resolved)


@dataclass
class LiteratureAgentConfig:
    controller: str = str(_LITERATURE_ROOT / "controller" / "literature_agent_full_end_to_end_v21_3_english_sanitizer.py")
    base_csv: str = str(_BACKEND_ROOT / "data" / "Perovskite_database_content_all_data.csv")
    ontology_path: str = str(_LITERATURE_ROOT / "config" / "perovskite_ontology_library_v19.json")
    work_dir: str = str(_RUNTIME_ROOT / "outputs")
    integration_out_dir: str = str(_RUNTIME_ROOT / "integration")
    model_script: str = str(_LITERATURE_ROOT / "models" / "pce_then_stability_same_approach.py")
    model_out_dir: str = str(_RUNTIME_ROOT / "models")
    knowledge_graph_out_dir: str = str(_RUNTIME_ROOT / "knowledge_graph")
    job_dir: str = str(_RUNTIME_ROOT / "jobs")
    python_executable: str = sys.executable
    llm_api_url: str = "http://localhost:11434/v1/chat/completions"
    llm_model: str = "qwen2.5:7b"
    vision_model: str = "qwen2.5vl:7b-q4_K_M"
    domain_profile: str = HALIDE_PV_DOMAIN_PROFILE
    knowledge_graph_builder: str = str(_LITERATURE_ROOT / "scripts" / "build_literature_knowledge_graph.py")

    def __post_init__(self) -> None:
        self.domain_profile = normalize_domain_profile(self.domain_profile)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LiteratureAgentConfig":
        configured_path = path or os.getenv("POLARIS_LITERATURE_CONFIG")
        config_path = (
            Path(configured_path).expanduser()
            if configured_path
            else _BACKEND_ROOT / "config" / "literature_agent.json"
        )
        payload: dict[str, Any] = {}
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            path_fields = {
                "controller", "base_csv", "ontology_path", "work_dir",
                "integration_out_dir", "model_script", "model_out_dir",
                "knowledge_graph_out_dir", "job_dir", "knowledge_graph_builder",
            }
            for field_name in path_fields:
                value = payload.get(field_name)
                if value and not Path(str(value)).expanduser().is_absolute():
                    payload[field_name] = str((config_path.parent / str(value)).resolve())
        env_map = {
            "domain_profile": "LITERATURE_AGENT_DOMAIN_PROFILE",
            "controller": "LITERATURE_AGENT_CONTROLLER",
            "base_csv": "LITERATURE_AGENT_BASE_CSV",
            "ontology_path": "LITERATURE_AGENT_ONTOLOGY",
            "work_dir": "LITERATURE_AGENT_WORK_DIR",
            "integration_out_dir": "LITERATURE_AGENT_INTEGRATION_DIR",
            "model_script": "LITERATURE_AGENT_MODEL_SCRIPT",
            "model_out_dir": "LITERATURE_AGENT_MODEL_DIR",
            "knowledge_graph_out_dir": "LITERATURE_AGENT_KG_DIR",
            "knowledge_graph_builder": "LITERATURE_AGENT_KG_BUILDER",
            "job_dir": "LITERATURE_AGENT_JOB_DIR",
        }
        for field_name, env_name in env_map.items():
            if os.getenv(env_name):
                payload[field_name] = os.environ[env_name]
        return cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})


@dataclass
class LiteratureJob:
    job_id: str
    stage: str
    status: str
    command: list[str]
    created_at: float
    updated_at: float
    pid: int | None = None
    return_code: int | None = None
    log_path: str | None = None
    request: dict[str, Any] = field(default_factory=dict)


class LiteratureAgentService:
    def __init__(self, config: LiteratureAgentConfig | None = None):
        self.config = config or LiteratureAgentConfig.load()
        self.job_dir = Path(self.config.job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def health(self) -> dict[str, Any]:
        paths = {
            "controller": self.config.controller,
            "base_csv": self.config.base_csv,
            "ontology_path": self.config.ontology_path,
            "knowledge_graph_builder": self.config.knowledge_graph_builder,
            "work_dir": self.config.work_dir,
            "all_records": str(Path(self.config.work_dir) / "csv" / "all_records.csv"),
            "summaries": str(Path(self.config.work_dir) / "paper_summaries_text"),
            "knowledge_graph": self.config.knowledge_graph_out_dir,
        }
        checks = {name: Path(value).exists() for name, value in paths.items()}
        return {
            "ok": checks["controller"] and checks["knowledge_graph_builder"] and checks["work_dir"],
            "service": "LiteratureAgent",
            "domain_profile": self.config.domain_profile,
            "domain_profiles": sorted(DOMAIN_PROFILES),
            "stages": sorted(STAGES),
            "extraction_profiles": sorted(EXTRACTION_PROFILES),
            "paths": paths,
            "path_checks": checks,
            "active_jobs": [job.job_id for job in self.list_jobs() if job.status == "running"],
        }

    def build_command(self, stage: str, request: dict[str, Any] | None = None) -> list[str]:
        if stage not in STAGES:
            raise ValueError(f"Unsupported stage: {stage}")
        request = request or {}
        domain_profile = normalize_domain_profile(
            request.get("domain_profile") or self.config.domain_profile
        )
        ontology_path = self.config.ontology_path
        if request.get("ontology_path") is not None:
            ontology_path = _resolve_request_ontology_path(request["ontology_path"])
        extraction_profile = str(request.get("extraction_profile", "none") or "none")
        if extraction_profile not in EXTRACTION_PROFILES:
            raise ValueError(f"Unsupported extraction_profile: {extraction_profile}")
        if domain_profile != HALIDE_PV_DOMAIN_PROFILE:
            if extraction_profile == PCE_EXTRACTION_PROFILE:
                raise ValueError(
                    "pce_stability_modeling is only available for domain_profile=halide_pv"
                )
            if stage == "integrate_and_model":
                raise ValueError(
                    "integrate_and_model is only available for domain_profile=halide_pv"
                )
        command = [
            self.config.python_executable,
            self.config.controller,
            "--pipeline_stage", stage,
            "--domain_profile", domain_profile,
            "--base_csv", self.config.base_csv,
            "--ontology_path", ontology_path,
            "--knowledge_graph_builder", self.config.knowledge_graph_builder,
            "--work_dir", self.config.work_dir,
            "--integration_out_dir", self.config.integration_out_dir,
            "--model_script", self.config.model_script,
            "--model_out_dir", self.config.model_out_dir,
            "--llm_api_url", self.config.llm_api_url,
            "--llm_model", self.config.llm_model,
            "--vision_model", self.config.vision_model,
            "--allow_embedded_reset", "0",
        ]
        if extraction_profile != "none":
            command += ["--extraction_profile", extraction_profile]
        if stage == "extract_batch":
            family_gating = (
                str(request.get("family_gating") or "strict")
                if domain_profile == HALIDE_PV_DOMAIN_PROFILE
                else NON_PV_FAMILY_GATING
            )
            command += [
                "--run_mode", str(request.get("run_mode", "expand")),
                "--max_papers", str(int(request.get("max_papers", 100))),
                "--full_literature_run",
                "--family_gating", family_gating,
                "--inline_vision", "0",
                "--vision_enable", "0",
                "--llm_cache_enable", "1",
                "--disable_google_drive", "1" if request.get("disable_google_drive", True) else "0",
                "--no_require_doi",
            ]
            if request.get("search_query"):
                command += ["--search_query", str(request["search_query"])]
            if request.get("search_plan_path"):
                command += ["--search_plan_path", str(request["search_plan_path"])]
        elif stage == "evidence_packet":
            query = str(request.get("query") or request.get("search_query") or "").strip()
            if not query:
                raise ValueError("Evidence packet requires a query")
            command += [
                "--evidence_query", query,
                "--evidence_limit", str(int(request.get("limit", 5))),
                "--evidence_max_chars", str(int(request.get("max_chars", 16000))),
                "--knowledge_graph_out_dir", self.config.knowledge_graph_out_dir,
            ]
            if request.get("evidence_out_dir"):
                command += ["--evidence_out_dir", str(request["evidence_out_dir"])]
        elif stage == "vision_pass":
            command += [
                "--vision_enable", "1",
                "--vision_only_max_papers", str(int(request.get("max_papers", 0))),
                "--vision_update_mode", str(request.get("vision_update_mode", "append")),
            ]
        elif stage == "integrate_and_model":
            command += ["--skip_literature_agent", "--no_require_doi"]
            if request.get("run_model"):
                command.append("--run_model")
        elif stage == "knowledge_graph":
            command += ["--skip_literature_agent", "--knowledge_graph_out_dir", self.config.knowledge_graph_out_dir]
        return command

    def start_stage(self, stage: str, request: dict[str, Any] | None = None, wait: bool = False) -> dict[str, Any]:
        request = request or {}
        job_id = f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        log_path = self.job_dir / f"{job_id}.log"
        command = self.build_command(stage, request)
        now = time.time()
        job = LiteratureJob(job_id, stage, "queued", command, now, now, log_path=str(log_path), request=request)
        self._write_job(job)
        with log_path.open("w", encoding="utf-8") as log:
            runner = Path(__file__).with_name("literature_agent_job_runner.py")
            process = subprocess.Popen(
                [self.config.python_executable, str(runner), str(self.job_dir / f"{job_id}.command.json")],
                cwd=str(Path(self.config.controller).parent),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        job.pid = process.pid
        job.status = "running"
        job.updated_at = time.time()
        self._write_job(job)
        if wait:
            return_code = process.wait()
            job.return_code = return_code
            job.status = "completed" if return_code == 0 else "failed"
            job.updated_at = time.time()
            self._write_job(job)
        return asdict(job)

    def job_status(self, job_id: str) -> dict[str, Any]:
        job = self._read_job(job_id)
        if job.status == "running" and job.pid and not _pid_exists(job.pid):
            return_code_path = self.job_dir / f"{job_id}.returncode"
            job.return_code = int(return_code_path.read_text()) if return_code_path.exists() else None
            job.status = "completed" if job.return_code in {None, 0} else "failed"
            job.updated_at = time.time()
            self._write_job(job)
        payload = asdict(job)
        payload["log_tail"] = self._log_tail(job.log_path)
        return payload

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self._read_job(job_id)
        if job.status == "running" and job.pid and _pid_exists(job.pid):
            if os.name == "nt":
                # The durable runner launches the controller as a child process.
                # Terminating only the runner leaves extraction and Ollama calls
                # alive, so Windows cancellation must close the complete tree.
                subprocess.run(
                    ["taskkill", "/PID", str(job.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            else:
                try:
                    os.killpg(os.getpgid(job.pid), signal.SIGTERM)
                except (AttributeError, ProcessLookupError):
                    os.kill(job.pid, signal.SIGTERM)
            job.status = "cancelled"
            job.updated_at = time.time()
            self._write_job(job)
        return asdict(job)

    def list_jobs(self) -> list[LiteratureJob]:
        jobs = []
        for path in sorted(self.job_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(LiteratureJob(**json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return jobs

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        terms = _evidence_terms(query)
        summaries_dir = Path(self.config.work_dir) / "paper_summaries_text"
        metadata = self._record_metadata()
        documents: list[dict[str, Any]] = []
        for path in summaries_dir.glob("*_summary.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            slug = path.name.removesuffix("_summary.txt")
            row = metadata.get(slug, {})
            title = row.get("Ref_title") or row.get("title") or _title_from_summary(text) or slug
            documents.append({
                "search_text": f"{title}\n{text}",
                "title_text": title,
                "payload": {
                        "paper_slug": slug,
                        "title": title,
                        "doi": row.get("Ref_DOI_number") or row.get("doi"),
                        "summary_excerpt": _excerpt(text, terms),
                        "summary_path": str(path),
                        "structured_json_path": str(Path(self.config.work_dir) / "json" / f"{slug}.json"),
                    },
            })
        ranked, _ = _rank_evidence_documents(query, documents, limit)
        ranked_slugs = {str(item.get("paper_slug") or "") for item in ranked}
        plan = build_literature_query_plan(query, self.config.ontology_path)
        contextual = []
        for document in documents:
            payload = dict(document.get("payload") or {})
            slug = str(payload.get("paper_slug") or "")
            if slug in ranked_slugs:
                continue
            allowed, diagnostics = query_plan_matches_text(
                plan,
                document.get("title_text"),
                document.get("search_text"),
            )
            if not allowed:
                continue
            normalized_text = f" {_normalized_phrase(document.get('search_text') or '')} "
            support_matches = [
                term for term in plan.get("support_terms") or []
                if (normalized := _normalized_phrase(term)) and f" {normalized} " in normalized_text
            ]
            payload.update({
                "score": round(10.0 * len(diagnostics.get("matched_groups") or []) + len(support_matches), 4),
                "query_coverage": 0.0,
                "matched_query_terms": support_matches,
                "matched_distinctive_terms": [],
                "distinctive_query_coverage": 0.0,
                "matched_query_phrases": [],
                "matched_ontology_groups": diagnostics.get("matched_groups") or [],
                "matched_ontology_aliases": diagnostics.get("matched_aliases") or {},
                "relevance_passed": True,
                "strong_relevance_passed": False,
                "evidence_scope": "ontology_matched_context",
            })
            contextual.append(payload)
        contextual.sort(key=lambda item: item.get("score", 0), reverse=True)
        return [*ranked, *contextual][:limit]

    def search_relationships(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        path = Path(self.config.knowledge_graph_out_dir) / "derived_views" / "learned_scientific_relationships.csv"
        if not path.exists():
            return []
        documents = []
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            for row in csv.DictReader(handle):
                text = " ".join(str(row.get(key, "")) for key in ("subject", "relationship", "object", "evidence", "evidence_text"))
                documents.append({
                    "search_text": text,
                    "title_text": " ".join(str(row.get(key, "")) for key in ("subject", "relationship", "object")),
                    "payload": {
                        "subject": row.get("subject"),
                        "relationship": row.get("relationship"),
                        "object": row.get("object"),
                        "paper_id": row.get("paper_id"),
                        "claim_id": row.get("claim_id"),
                        "evidence": row.get("evidence") or row.get("evidence_text"),
                    },
                })
        ranked, _ = _rank_evidence_documents(query, documents, limit)
        return ranked

    def evidence_packet(self, query: str, limit: int = 5) -> dict[str, Any]:
        papers = self.search(query, limit=limit)
        relationships = self.search_relationships(query, limit=limit)
        lines = ["Evidence-grounded LiteratureAgent packet:"]
        strong_evidence_count = sum(
            1 for item in papers + relationships if item.get("strong_relevance_passed")
        )
        if strong_evidence_count == 0:
            lines.append(
                "No strong direct local support matched the full query. The items below are related background only; "
                "do not present them as proof of the combined claim."
            )
        for index, paper in enumerate(papers, 1):
            lines.append(
                f"\nPaper {index}: {paper['title']}\n"
                f"DOI: {paper.get('doi') or 'not available'} | paper_slug: {paper['paper_slug']}\n"
                f"Summary evidence: {paper['summary_excerpt']}"
            )
        if relationships:
            lines.append("\nLearned scientific relationships:")
            for index, rel in enumerate(relationships, 1):
                lines.append(
                    f"{index}. {rel.get('subject')} -> {rel.get('relationship')} -> {rel.get('object')} "
                    f"[paper={rel.get('paper_id') or 'unknown'}, claim={rel.get('claim_id') or 'unknown'}]"
                )
        return _enrich_evidence_packet({
            "query": query,
            "papers": papers,
            "relationships": relationships,
            "formatted_context": "\n".join(lines)[:16000],
            "provenance": {
                "work_dir": self.config.work_dir,
                "knowledge_graph_dir": self.config.knowledge_graph_out_dir,
                "paper_count": len(papers),
                "relationship_count": len(relationships),
                "relevant_evidence_count": len(papers) + len(relationships),
                "strong_evidence_count": strong_evidence_count,
            },
        })

    def write_evidence_packet(self, query: str, limit: int = 5, max_chars: int = 16000) -> dict[str, Any]:
        """Create an auditable evidence packet with the controller when available.

        The in-process packet is still used as a fallback so hypothesis calls do
        not fail just because the standalone controller is unavailable.
        """
        out_dir = Path(self.config.work_dir) / "evidence_packets"
        try:
            job = self.start_stage(
                "evidence_packet",
                {
                    "query": query,
                    "limit": limit,
                    "max_chars": max_chars,
                    "evidence_out_dir": str(out_dir),
                    "extraction_profile": "hypothesis_support",
                },
                wait=True,
            )
            if job.get("return_code") not in {None, 0}:
                raise RuntimeError(f"evidence_packet controller returned {job.get('return_code')}")
            packet_path = _latest_evidence_packet(out_dir)
            if packet_path:
                packet = json.loads(packet_path.read_text(encoding="utf-8"))
                packet.setdefault("provenance", {})["controller_packet_path"] = str(packet_path)
                packet = _enrich_evidence_packet(packet, max_chars=max_chars)
                if not packet.get("claims"):
                    local_packet = self.evidence_packet(query, limit=limit)
                    if local_packet.get("claims"):
                        local_packet.setdefault("provenance", {})["controller_packet_path"] = str(packet_path)
                        local_packet["provenance"]["controller_packet_replaced_reason"] = (
                            "Controller literal-term retrieval returned no claims; used ontology-matched local summaries."
                        )
                        packet = local_packet
                packet_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
                packet_path.with_suffix(".txt").write_text(
                    str(packet.get("formatted_context") or "").rstrip() + "\n", encoding="utf-8"
                )
                return packet
        except Exception as exc:
            packet = self.evidence_packet(query, limit=limit)
            packet.setdefault("provenance", {})["controller_fallback_reason"] = repr(exc)
            packet["formatted_context"] = packet["formatted_context"][:max_chars]
            return packet
        return self.evidence_packet(query, limit=limit)

    def active_retrieval_job_for_query(self, query: str) -> dict[str, Any] | None:
        normalized = _normalize_query(query)
        for job in self.list_jobs():
            if job.stage != "extract_batch" or job.status not in {"queued", "running"}:
                continue
            if _normalize_query(str(job.request.get("search_query", ""))) == normalized:
                return asdict(job)
        return None

    def start_hypothesis_support_retrieval(
        self, query: str, max_papers: int = 3, attempt: int = 1
    ) -> dict[str, Any]:
        plan = build_literature_query_plan(query, self.config.ontology_path)
        variants = list(plan.get("search_variants") or [_build_external_search_query(query)])
        variant_index = min(max(1, int(attempt)) - 1, len(variants) - 1)
        search_query = variants[variant_index]
        plan_dir = Path(self.config.work_dir) / "search_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_hash = hashlib.sha256(query.encode("utf-8", errors="ignore")).hexdigest()[:16]
        plan_path = plan_dir / f"{plan_hash}_search_plan.json"
        plan["selected_attempt"] = int(attempt)
        plan["selected_query"] = search_query
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        existing = self.active_retrieval_job_for_query(search_query)
        if existing:
            existing["already_running"] = True
            existing["search_plan_path"] = str(plan_path)
            return existing
        return self.start_stage(
            "extract_batch",
            {
                "search_query": search_query,
                "source_query": query,
                "search_plan_path": str(plan_path),
                "retrieval_attempt": int(attempt),
                "max_papers": max(1, min(int(max_papers), 10)),
                "extraction_profile": "hypothesis_support",
                "family_gating": "off",
                "disable_google_drive": True,
            },
            wait=False,
        )

    def evidence_context(self, query: str, limit: int = 5, max_chars: int = 12000) -> str:
        packet = self.evidence_packet(query, limit=limit)
        if not packet["papers"] and not packet["relationships"]:
            return "No matching locally mined LiteratureAgent evidence was found."
        return packet["formatted_context"][:max_chars]

    def execute_task(self, task: AgentTask) -> AgentResult:
        actions = {
            "health": lambda: self.health(),
            "artifacts": lambda: self.artifact_manifest(),
            "search": lambda: self.search(str(task.payload.get("query", "")), int(task.payload.get("limit", 8))),
            "evidence_packet": lambda: self.write_evidence_packet(
                str(task.payload.get("query", "")),
                int(task.payload.get("limit", 5)),
                int(task.payload.get("max_chars", 16000)),
            ),
            "retrieve_for_hypothesis": lambda: self.start_hypothesis_support_retrieval(
                str(task.payload.get("query", "")),
                int(task.payload.get("max_papers", 3)),
                int(task.payload.get("attempt", 1)),
            ),
            "job_status": lambda: self.job_status(str(task.payload["job_id"])),
            "cancel_job": lambda: self.cancel_job(str(task.payload["job_id"])),
            "start_stage": lambda: self.start_stage(
                str(task.payload["stage"]), dict(task.payload.get("request", {})), bool(task.payload.get("wait", False))
            ),
        }
        if task.action not in actions:
            return AgentResult(task.task_id, "literature_agent", task.action, "failed", error="Unsupported action")
        data = actions[task.action]()
        return AgentResult(
            task_id=task.task_id,
            agent_id="literature_agent",
            action=task.action,
            status="completed",
            data=data,
            artifacts=self.artifact_manifest() if task.action in {"evidence_packet", "start_stage"} else {},
        )

    def artifact_manifest(self) -> dict[str, Any]:
        work = Path(self.config.work_dir)
        paths = {
            "all_records_csv": work / "csv" / "all_records.csv",
            "paper_registry": work / "paper_registry.json",
            "paper_type_gate_report": work / "paper_type_gate_report.csv",
            "paper_summaries_text": work / "paper_summaries_text",
            "paper_summaries_json": work / "paper_summaries_json",
            "structured_json": work / "json",
            "figure_reports": work / "figure_reports",
            "integration": Path(self.config.integration_out_dir),
            "models": Path(self.config.model_out_dir),
            "knowledge_graph": Path(self.config.knowledge_graph_out_dir),
        }
        return {
            name: {"path": str(path), "exists": path.exists(), "files": _file_count(path)}
            for name, path in paths.items()
        }

    def _record_metadata(self) -> dict[str, dict[str, str]]:
        registry_path = Path(self.config.work_dir) / "paper_registry.json"
        if registry_path.exists():
            try:
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
                papers = payload.get("papers", {}) if isinstance(payload, dict) else {}
                if isinstance(papers, dict):
                    return {
                        str(record.get("slug") or key): record
                        for key, record in papers.items()
                        if isinstance(record, dict) and (record.get("slug") or key)
                    }
            except Exception:
                pass
        csv_path = Path(self.config.work_dir) / "csv" / "all_records.csv"
        if not csv_path.exists():
            return {}
        result: dict[str, dict[str, str]] = {}
        with csv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            for row in csv.DictReader(handle):
                slug = row.get("paper_slug") or row.get("_paper_slug") or row.get("Ref_internal_sample_id")
                if slug:
                    result[str(slug)] = row
        return result

    def _write_job(self, job: LiteratureJob) -> None:
        (self.job_dir / f"{job.job_id}.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        (self.job_dir / f"{job.job_id}.command.json").write_text(
            json.dumps({"job_id": job.job_id, "command": job.command, "job_dir": str(self.job_dir)}, indent=2),
            encoding="utf-8",
        )

    def _read_job(self, job_id: str) -> LiteratureJob:
        path = self.job_dir / f"{job_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown LiteratureAgent job: {job_id}")
        return LiteratureJob(**json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _log_tail(log_path: str | None, lines: int = 30) -> str:
        if not log_path or not Path(log_path).exists():
            return ""
        return "\n".join(Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _latest_evidence_packet(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    packets = sorted(out_dir.glob("*_evidence_packet.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return packets[0] if packets else None


def _evidence_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9+]{1,}", str(query or ""))
    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        term = raw.lower().strip("+")
        keep_short = any(char.isdigit() for char in term) or raw.isupper()
        if (len(term) < 3 and not keep_short) or term in EVIDENCE_QUERY_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _normalized_phrase(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value or ""))
    value = (
        value.replace("α", " alpha ").replace("β", " beta ").replace("γ", " gamma ")
        .replace("δ", " delta ").replace("Δ", " delta ")
    )
    value = re.sub(r"[^A-Za-z0-9+]+", " ", value.lower())
    return " ".join(value.split())


def _phrase_present(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalized_phrase(text)} "
    normalized_phrase = _normalized_phrase(phrase)
    return bool(normalized_phrase and f" {normalized_phrase} " in normalized_text)


def _ontology_alias_groups(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    groups: list[dict[str, Any]] = []

    def walk(value: Any, trail: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, trail + [str(key)])
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            canonical = trail[-1] if trail else "concept"
            aliases = list(dict.fromkeys([canonical, *[str(item) for item in value]]))
            groups.append({"concept": canonical, "aliases": aliases, "ontology_path": ".".join(trail)})
        elif isinstance(value, list):
            for child in value:
                walk(child, trail)

    walk(payload, [])
    return groups


def build_literature_query_plan(query: str, ontology_path: str | Path) -> dict[str, Any]:
    """Build a compact, auditable and domain-configurable retrieval plan."""
    query = str(query or "").strip()
    matched_groups: list[dict[str, Any]] = []
    seen_concepts: set[str] = set()
    for group in _ontology_alias_groups(ontology_path):
        matched = [alias for alias in group["aliases"] if _phrase_present(query, alias)]
        if not matched:
            continue
        concept_key = _normalized_phrase(group["concept"])
        if concept_key in QUERY_GENERIC_CONCEPTS:
            continue
        if all(_normalized_phrase(alias) in QUERY_GENERIC_MATCHES for alias in matched):
            continue
        if concept_key in seen_concepts:
            continue
        seen_concepts.add(concept_key)
        expanded = sorted(
            group["aliases"],
            key=lambda alias: (
                0 if alias in matched else 1,
                0 if (" " in alias or "-" in alias or any(ch.isdigit() for ch in alias)) else 1,
                len(alias),
            ),
        )
        aliases = list(dict.fromkeys([*matched, *expanded]))[:12]
        representative = next(
            (alias for alias in matched if len(_normalized_phrase(alias)) > 3),
            next((alias for alias in aliases if len(_normalized_phrase(alias)) > 3), group["concept"]),
        )
        matched_groups.append({
            "concept": group["concept"],
            "aliases": aliases,
            "matched_aliases": matched,
            "representative": representative,
            "ontology_path": group["ontology_path"],
        })

    # Prefer the most specific matched ontology concepts and remove groups whose
    # matched aliases are fully contained by a stronger group.
    matched_groups.sort(
        key=lambda group: (
            -max((len(_normalized_phrase(alias)) for alias in group["matched_aliases"]), default=0),
            len(group.get("aliases") or []),
        ),
    )
    required_groups: list[dict[str, Any]] = []
    for group in matched_groups:
        matched_norm = {_normalized_phrase(alias) for alias in group["matched_aliases"]}
        if any(matched_norm and matched_norm <= {
            _normalized_phrase(alias) for alias in kept["matched_aliases"]
        } for kept in required_groups):
            continue
        required_groups.append(group)
        if len(required_groups) >= 5:
            break

    terms = [
        term for term in _evidence_terms(query)
        if term not in QUERY_GENERIC_TERMS
        and not any(_phrase_present(alias, term) for group in required_groups for alias in group["aliases"])
    ]
    if not required_groups:
        fallback_terms = [term for term in terms if len(term) >= 4 or any(ch.isdigit() for ch in term)][:5]
        required_groups = [
            {"concept": term, "aliases": [term], "matched_aliases": [term], "representative": term, "ontology_path": "fallback"}
            for term in fallback_terms
        ]

    representatives = [str(group["representative"]) for group in required_groups]

    # Crossref keyword search is sensitive to nomenclature. Rotate close ontology
    # aliases between attempts while keeping the same concept-level hard gate.
    alternate_representatives = []
    for group, representative in zip(required_groups, representatives):
        representative_norm = _normalized_phrase(representative)
        alternatives = []
        if re.search(r"\d", representative_norm):
            alternate_representatives.append(representative)
            continue
        for alias in group.get("aliases") or []:
            alias_norm = _normalized_phrase(alias)
            if not alias_norm or alias_norm == representative_norm:
                continue
            if bool(re.search(r"\d", representative_norm)) != bool(re.search(r"\d", alias_norm)):
                continue
            similarity = difflib.SequenceMatcher(None, representative_norm, alias_norm).ratio()
            if similarity >= 0.72:
                alternatives.append((similarity, len(alias_norm), str(alias)))
        alternatives.sort(key=lambda item: (-item[0], item[1], item[2].lower()))
        alternate_representatives.append(alternatives[0][2] if alternatives else representative)

    support_terms = terms[:8]
    variant_candidates = [
        " ".join([*representatives, *support_terms[:5]]),
        " ".join([*alternate_representatives, *support_terms[2:7]]),
        " ".join([*representatives, *support_terms[:2]]),
    ]
    variants = list(dict.fromkeys(" ".join(value.split()) for value in variant_candidates if value.strip()))
    min_required_groups = 1 if len(required_groups) <= 1 else 2
    return {
        "version": "polaris_query_plan_v1",
        "source_query": query,
        "required_groups": required_groups,
        "minimum_required_groups": min_required_groups,
        "support_terms": support_terms,
        "search_variants": variants or [query],
    }


def query_plan_matches_text(plan: dict[str, Any], *values: Any) -> tuple[bool, dict[str, Any]]:
    groups = list(plan.get("required_groups") or [])
    if not groups:
        return True, {"matched_groups": [], "minimum_required_groups": 0}
    text = f" {_normalized_phrase(' '.join(str(value or '') for value in values))} "
    matched_groups = []
    matched_aliases: dict[str, list[str]] = {}
    for group in groups:
        aliases = [
            str(alias) for alias in group.get("aliases") or []
            if (normalized := _normalized_phrase(alias)) and f" {normalized} " in text
        ]
        if aliases:
            concept = str(group.get("concept") or "concept")
            matched_groups.append(concept)
            matched_aliases[concept] = aliases[:6]
    minimum = max(1, int(plan.get("minimum_required_groups") or 1))
    diagnostics = {
        "matched_groups": matched_groups,
        "matched_aliases": matched_aliases,
        "required_group_count": len(groups),
        "minimum_required_groups": minimum,
    }
    return len(matched_groups) >= minimum, diagnostics


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or not str(value).strip():
        return []
    return [str(value).strip()]


def _summary_payload_for_paper(paper: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(paper.get("summary_path") or ""))
    candidates = []
    if summary_path:
        candidates.append(
            summary_path.parent.parent / "paper_summaries_json" / f"{paper.get('paper_slug')}_summary.json"
        )
    structured = str(paper.get("structured_json_path") or "").strip()
    if structured:
        candidates.append(Path(structured))
    for path in candidates:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
    return {}


def _enrich_evidence_packet(packet: dict[str, Any], max_chars: int = 16000) -> dict[str, Any]:
    papers = list(packet.get("papers") or [])
    relationships = list(packet.get("relationships") or [])
    claims: list[dict[str, Any]] = []
    for paper in papers:
        if not paper.get("relevance_passed") and not paper.get("strong_relevance_passed"):
            continue
        payload = _summary_payload_for_paper(paper)
        summary = payload.get("paper_summary") if isinstance(payload, dict) else {}
        summary = summary if isinstance(summary, dict) else {}
        strong_relevance = bool(paper.get("strong_relevance_passed"))
        relevance = str(summary.get("query_relevance") or "indirect_context")
        if not strong_relevance:
            relevance = "partial_context_only"
        source_id = str(paper.get("doi") or paper.get("paper_slug") or "unknown")
        limitations = _as_text_list(summary.get("limitations_and_uncertainties"))[:3]
        abstract_only = any("abstract" in item.lower() for item in limitations)
        for evidence in _as_text_list(summary.get("direct_evidence"))[:4]:
            claims.append({
                "claim_id": f"E{len(claims) + 1}",
                "claim": evidence,
                "evidence_type": (
                    "contextual_abstract_evidence" if abstract_only else
                    "direct_reported_evidence" if strong_relevance else
                    "contextual_reported_evidence"
                ),
                "query_relevance": relevance,
                "source_id": source_id,
                "doi": paper.get("doi"),
                "paper_slug": paper.get("paper_slug"),
                "title": paper.get("title"),
                "limitations": limitations,
            })
        for interpretation in _as_text_list(summary.get("author_interpretation"))[:2]:
            claims.append({
                "claim_id": f"E{len(claims) + 1}",
                "claim": interpretation,
                "evidence_type": "author_interpretation" if strong_relevance else "contextual_author_interpretation",
                "query_relevance": relevance,
                "source_id": source_id,
                "doi": paper.get("doi"),
                "paper_slug": paper.get("paper_slug"),
                "title": paper.get("title"),
                "limitations": limitations,
            })
    for relationship in relationships:
        if not relationship.get("strong_relevance_passed"):
            continue
        source_id = str(relationship.get("paper_id") or relationship.get("claim_id") or "unknown")
        claims.append({
            "claim_id": f"E{len(claims) + 1}",
            "claim": (
                f"{relationship.get('subject')} {relationship.get('relationship')} {relationship.get('object')}"
            ).strip(),
            "evidence_type": "knowledge_graph_relationship",
            "query_relevance": "direct_or_contextual",
            "source_id": source_id,
            "paper_slug": relationship.get("paper_id"),
            "limitations": [],
        })

    lines = ["Evidence-grounded LiteratureAgent packet:"]
    if claims:
        lines.append(
            "Use the evidence claims below when directly relevant. Cite the claim ID and DOI or paper_slug. "
            "Keep author interpretation separate from directly reported evidence. Claims labeled "
            "partial_context_only support only part of the question and must not be presented as proof of the "
            "full proposed mechanism."
        )
        for claim in claims:
            lines.append(
                f"\n[{claim['claim_id']}] {claim['evidence_type']} | relevance={claim['query_relevance']} | "
                f"source={claim['source_id']}\nClaim: {claim['claim']}"
            )
            if claim.get("limitations"):
                lines.append("Limitations: " + "; ".join(claim["limitations"]))
    else:
        lines.append(
            "No strong direct local support matched the full query. Do not present related background as proof "
            "and do not invent citations."
        )

    packet["claims"] = claims
    packet["formatted_context"] = "\n".join(lines)[:max_chars]
    provenance = packet.setdefault("provenance", {})
    provenance["claim_count"] = len(claims)
    provenance["claim_ids"] = [claim["claim_id"] for claim in claims]
    provenance["source_ids"] = list(dict.fromkeys(claim["source_id"] for claim in claims))
    return packet


def _rank_evidence_documents(
    query: str, documents: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terms = _evidence_terms(query)
    if not terms or not documents:
        return [], {"query_terms": terms, "distinctive_terms": [], "minimum_coverage": 0.0}

    prepared = []
    for document in documents:
        search_text = re.sub(r"[-–—/]", " ", str(document.get("search_text") or "").lower())
        tokens = re.findall(r"[a-z0-9][a-z0-9+]{1,}", search_text)
        prepared.append((document, tokens, set(tokens)))

    document_count = len(prepared)
    document_frequency = {
        term: sum(1 for _, _, token_set in prepared if term in token_set)
        for term in terms
    }
    weights = {
        term: math.log((document_count + 1.0) / (document_frequency[term] + 1.0)) + 1.0
        for term in terms
    }
    distinctive_count = max(1, min(8, (len(terms) + 1) // 2))
    distinctive_terms = sorted(terms, key=lambda term: (weights[term], len(term)), reverse=True)[:distinctive_count]
    total_weight = sum(weights.values()) or 1.0
    minimum_matches = 1 if len(terms) <= 2 else 2
    minimum_coverage = 0.0 if len(terms) <= 2 else 0.16
    minimum_distinctive_matches = 1 if len(terms) <= 4 else 2

    phrases: list[str] = []
    for size in (3, 2):
        for index in range(len(terms) - size + 1):
            phrase = " ".join(terms[index:index + size])
            if phrase not in phrases:
                phrases.append(phrase)

    ranked = []
    for document, tokens, token_set in prepared:
        matched_terms = [term for term in terms if term in token_set]
        matched_weight = sum(weights[term] for term in matched_terms)
        coverage = matched_weight / total_weight
        distinctive_matches = [term for term in distinctive_terms if term in token_set]
        normalized_text = " ".join(tokens)
        matched_phrases = [
            phrase for phrase in phrases
            if phrase in normalized_text and any(term in distinctive_terms for term in phrase.split())
        ]
        if not (
            len(matched_terms) >= minimum_matches
            and coverage >= minimum_coverage
            and (len(distinctive_matches) >= minimum_distinctive_matches or len(terms) <= 2)
        ):
            continue

        frequency_score = sum(
            weights[term] * (1.0 + math.log1p(min(tokens.count(term), 3)))
            for term in matched_terms
        )
        title_tokens = set(re.findall(
            r"[a-z0-9][a-z0-9+]{1,}", re.sub(r"[-–—/]", " ", str(document.get("title_text") or "").lower())
        ))
        title_bonus = sum(weights[term] for term in matched_terms if term in title_tokens) * 1.5
        phrase_bonus = sum(2.0 + 0.5 * len(phrase.split()) for phrase in matched_phrases)
        score = frequency_score + title_bonus + phrase_bonus + coverage * 5.0
        distinctive_coverage = len(distinctive_matches) / max(1, len(distinctive_terms))
        strong_relevance = (
            len(terms) <= 2
            or (
                len(distinctive_matches) >= 2
                and distinctive_coverage >= 0.5
                and coverage >= 0.25
            )
        )
        payload = dict(document.get("payload") or {})
        payload.update({
            "score": round(score, 4),
            "query_coverage": round(coverage, 4),
            "matched_query_terms": matched_terms,
            "matched_distinctive_terms": distinctive_matches,
            "distinctive_query_coverage": round(distinctive_coverage, 4),
            "matched_query_phrases": matched_phrases[:8],
            "relevance_passed": True,
            "strong_relevance_passed": strong_relevance,
        })
        ranked.append((score, coverage, payload))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    diagnostics = {
        "query_terms": terms,
        "distinctive_terms": distinctive_terms,
        "term_document_frequency": document_frequency,
        "minimum_matches": minimum_matches,
        "minimum_coverage": minimum_coverage,
        "minimum_distinctive_matches": minimum_distinctive_matches,
        "documents_considered": document_count,
        "documents_relevant": len(ranked),
        "documents_strongly_relevant": sum(
            1 for _, _, payload in ranked if payload.get("strong_relevance_passed")
        ),
    }
    return [payload for _, _, payload in ranked[:limit]], diagnostics


def _normalize_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+\-]{2,}", str(query).lower()))


def _build_external_search_query(query: str, max_terms: int = 18) -> str:
    """Turn a conversational request into a concise, domain-neutral search query."""
    terms = _evidence_terms(query)
    if not terms:
        return str(query or "").strip()
    return " ".join(terms[:max(4, int(max_terms))])


def _title_from_summary(text: str) -> str | None:
    for line in text.splitlines()[:12]:
        cleaned = line.strip().strip("#").strip()
        if cleaned and len(cleaned) > 8 and not cleaned.lower().startswith(("paper summary", "executive summary")):
            return cleaned
    return None


def _excerpt(text: str, terms: list[str], radius: int = 900) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    start = max(0, (min(positions) if positions else 0) - 200)
    return " ".join(text[start : start + radius].split())
