#!/usr/bin/env python
"""Run controlled, resumable LiteratureAgent expansion batches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_BATCHES = {"pilot": 100, "scale": 750, "full": 1000}
REPO_ROOT = Path(__file__).resolve().parents[4]
TARGET_RICH_QUERY = (
    '"perovskite solar cell" experimental device PCE J-V Voc Jsc fill factor '
    "stability T80 T95 retention lifetime aging humidity illumination encapsulation"
)


def run(command: list[str], dry_run: bool) -> None:
    print("\n>", subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, check=True)


def find_csv(folder: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_previous_updated_csv(campaign_root: Path, current_batch: Path) -> Path | None:
    candidates = [
        path
        for path in campaign_root.glob("*/integration/updated_perovskite_database_with_literature_agent.csv")
        if current_batch not in path.parents
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def registry_paper_count(work_dir: Path) -> int | None:
    registry = work_dir / "paper_registry.json"
    if not registry.exists():
        return None
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ["papers", "records", "items"]:
                if isinstance(payload.get(key), (list, dict)):
                    return len(payload[key])
            return len(payload)
        if isinstance(payload, list):
            return len(payload)
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "scale", "full", "audit-only"], required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--target-total", type=int, default=5000)
    parser.add_argument(
        "--search-query",
        nargs="+",
        default=[TARGET_RICH_QUERY],
        help="Target-rich Crossref/RSS query words for this batch. Multiple arguments are joined safely.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=REPO_ROOT / "expansion_campaign",
    )
    parser.add_argument("--controller", type=Path, default=Path(__file__).parents[1] / "literature_agent_full_end_to_end_v21_3_english_sanitizer.py")
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=REPO_ROOT / "data" / "Perovskite_database_content_all_data.csv",
    )
    parser.add_argument(
        "--ontology-path",
        type=Path,
        default=REPO_ROOT / "config" / "perovskite_ontology_library_v19.json",
    )
    parser.add_argument("--oauth-secrets", type=Path, help="Optional Google Drive OAuth client-secrets JSON.")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "lit_outputs")
    parser.add_argument("--run-model-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.search_query = " ".join(args.search_query).strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = args.batch_size or DEFAULT_BATCHES.get(args.stage, 0)
    batch_name = f"{args.stage}_{batch_size}_{timestamp}"
    batch_dir = args.campaign_root / batch_name
    integration_dir = batch_dir / "integration"
    qa_dir = batch_dir / "qa"
    batch_dir.mkdir(parents=True, exist_ok=True)
    papers_before = registry_paper_count(args.work_dir)
    audit_baseline = find_previous_updated_csv(args.campaign_root, batch_dir) or args.base_csv

    common = [
        sys.executable,
        str(args.controller),
        "--base_csv", str(args.base_csv),
        "--ontology_path", str(args.ontology_path),
        "--work_dir", str(args.work_dir),
        "--integration_out_dir", str(integration_dir),
        "--run_mode", "expand",
        "--family_gating", "strict",
        "--llm_cache_enable", "1",
        "--vision_enable", "0",
        "--inline_vision", "0",
        "--figure_report_enable", "1",
        "--missing_field_recovery", "0",
        "--allow_embedded_reset", "0",
        "--disable_google_drive", "1",
        "--no_require_doi",
    ]
    if args.oauth_secrets:
        common += ["--google_drive_oauth_client_secrets", str(args.oauth_secrets)]

    if args.stage != "audit-only":
        extraction = common + [
            "--pipeline_stage", "extract_batch",
            "--full_literature_run",
            "--max_papers", str(batch_size),
            "--drive_process_all_files", "0",
            "--google_drive_max_files_per_run", str(batch_size),
            "--crossref_max_pages", str(max(10, batch_size // 20)),
            "--search_query", args.search_query,
        ]
        run(extraction, args.dry_run)

        integration = common + ["--pipeline_stage", "integrate_and_model", "--skip_literature_agent"]
        if args.run_model_check:
            integration.append("--run_model")
        run(integration, args.dry_run)

    updated = integration_dir / "updated_perovskite_database_with_literature_agent.csv"
    raw = args.work_dir / "csv" / "all_records.csv"
    accepted = find_csv(integration_dir, ["*accepted*.csv"])
    rejected = find_csv(integration_dir, ["*rejected*.csv"])
    audit = [
        sys.executable,
        str(Path(__file__).with_name("audit_expansion_batch.py")),
        "--batch-name", batch_name,
        "--base-csv", str(audit_baseline),
        "--updated-csv", str(updated),
        "--raw-literature-csv", str(raw),
        "--paper-type-report", str(args.work_dir / "paper_type_gate_report.csv"),
        "--out-dir", str(qa_dir),
    ]
    if accepted:
        audit += ["--accepted-csv", str(accepted)]
    if rejected:
        audit += ["--rejected-csv", str(rejected)]
    run(audit, args.dry_run)

    manifest = {
        "batch_name": batch_name,
        "stage": args.stage,
        "batch_size": batch_size,
        "created": datetime.now().isoformat(),
        "work_dir": str(args.work_dir),
        "integration_dir": str(integration_dir),
        "qa_dir": str(qa_dir),
        "vision_inline": False,
        "family_gating": "strict",
        "dry_run": args.dry_run,
        "target_total_papers": args.target_total,
        "registered_papers_before": papers_before,
        "registered_papers_after": registry_paper_count(args.work_dir),
        "audit_baseline_csv": str(audit_baseline),
        "search_query": args.search_query,
    }
    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nBatch folder: {batch_dir}")


if __name__ == "__main__":
    main()
