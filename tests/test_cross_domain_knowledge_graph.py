import csv
import json
import subprocess
import sys
from pathlib import Path


def test_generic_battery_record_builds_evidence_grounded_graph(tmp_path):
    backend_root = Path(__file__).resolve().parents[1]
    records_path = tmp_path / "battery_records.csv"
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "graph"
    work_dir.mkdir()
    summaries = work_dir / "paper_summaries_json"
    summaries.mkdir()
    (summaries / "battery_paper_summary.json").write_text(
        json.dumps(
            {
                "paper_summary": {
                    "primary_findings": ["The electrolyte increased ionic conductivity."],
                    "best_reported_performance": {"pce": 99.0},
                }
            }
        ),
        encoding="utf-8",
    )

    record = {
        "_domain_profile": "general_science",
        "_domain_pack_id": "battery_materials",
        "_domain_pack_version": "v2026.1",
        "_domain_pack_file": "battery_materials.json",
        "Ref_internal_sample_id": "battery_paper_device_1",
        "Ref_DOI_number": "10.0000/battery-smoke",
        "Domain_system_or_subject": "lithium metal battery",
        "Domain_materials_or_entities": json.dumps(
            ["lithium metal", "solid polymer electrolyte"]
        ),
        "Domain_methods": "electrochemical impedance spectroscopy",
        "Domain_conditions": "25 degrees C",
        "Domain_measurements": json.dumps(
            [
                {
                    "name": "ionic conductivity",
                    "value": 1.2,
                    "unit": "mS cm-1",
                    "conditions": "25 degrees C",
                }
            ]
        ),
        "Domain_electrolyte_type": "solid polymer electrolyte",
        "_evidence_Domain_system_or_subject": "We assembled lithium metal batteries.",
        "_evidence_Domain_measurements": "Ionic conductivity was 1.2 mS cm-1 at 25 degrees C.",
        "_evidence_Domain_electrolyte_type": "A solid polymer electrolyte was used.",
    }
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)

    result = subprocess.run(
        [
            sys.executable,
            str(backend_root / "literature_agent" / "scripts" / "build_literature_knowledge_graph.py"),
            "--records",
            str(records_path),
            "--ontology",
            str(backend_root / "literature_agent" / "config" / "domain_packs" / "general_science.json"),
            "--work-dir",
            str(work_dir),
            "--out",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert "Knowledge graph written" in result.stdout
    with (out_dir / "neo4j_import" / "nodes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        nodes = list(csv.DictReader(handle))
    with (out_dir / "neo4j_import" / "relationships.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        relationships = list(csv.DictReader(handle))

    assert any(
        "StudiedSystem" in node[":LABEL"] and node["name"] == "lithium metal battery"
        for node in nodes
    )
    assert any(
        "DomainMeasurement" in node[":LABEL"] and node["name"] == "ionic conductivity"
        for node in nodes
    )
    assert any(
        "DomainAttribute" in node[":LABEL"] and node["name"] == "solid polymer electrolyte"
        for node in nodes
    )
    assert any(edge[":TYPE"] == "HAS_DOMAIN_ATTRIBUTE" for edge in relationships)
    assert any(edge[":TYPE"] == "SUPPORTS_ENTITY_OR_MEASUREMENT" for edge in relationships)
    assert not any("Device" in node[":LABEL"].split(";") for node in nodes)
    assert not any("PerformanceMeasurement" in node[":LABEL"].split(";") for node in nodes)
    assert not any(edge[":TYPE"] == "REPORTS_SUMMARY_PERFORMANCE" for edge in relationships)
    paper_properties = [
        json.loads(node["properties_json"])
        for node in nodes
        if "Paper" in node[":LABEL"].split(";")
    ]
    assert paper_properties[0]["domain_pack_id"] == "battery_materials"
    assert paper_properties[0]["domain_pack_version"] == "v2026.1"
