import csv
import json

from app.services.literature_kg_retrieval import (
    build_hypothesis_kg_context,
    retrieve_literature_kg_claims,
)


def _write_graph(root):
    neo4j = root / "neo4j"
    neo4j.mkdir(parents=True)
    nodes = [
        ("snapshot:1", "DatasetSnapshot", {"snapshot_id": "test-graph"}),
        (
            "paper:1",
            "Paper",
            {"doi": "10.1000/test", "title": "2D 3D perovskite interfaces"},
        ),
        (
            "claim:1",
            "ScientificClaim",
            {
                "claim_text": "A 2D interlayer was associated with improved stability.",
                "predicate": "associated_with",
                "causal_status": "reported_association",
                "support_status": "evidence_linked",
                "doi": "10.1000/test",
            },
        ),
        (
            "claim:2",
            "ScientificClaim",
            {
                "claim_text": "A 2D interlayer was associated with improved stability.",
                "predicate": "associated_with",
                "causal_status": "reported_association",
                "support_status": "evidence_linked",
                "doi": "10.1000/test",
            },
        ),
        ("entity:1", "ScientificEntity", {"label": "2D interlayer"}),
        ("entity:2", "ScientificEntity", {"label": "device stability"}),
        (
            "evidence:1",
            "Evidence",
            {"evidence_text": "The treated devices retained stability."},
        ),
    ]
    with (neo4j / "nodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id:ID", ":LABEL", "properties_json"))
        for node_id, label, properties in nodes:
            writer.writerow((node_id, label, json.dumps(properties)))
    relationships = [
        ("paper:1", "REPORTS_CLAIM", "claim:1"),
        ("paper:1", "REPORTS_CLAIM", "claim:2"),
        ("claim:1", "SUPPORTED_BY", "evidence:1"),
        ("claim:1", "HAS_SUBJECT", "entity:1"),
        ("claim:1", "HAS_OBJECT", "entity:2"),
    ]
    with (neo4j / "relationships.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (":START_ID", ":TYPE", ":END_ID", "relationship_id", "properties_json")
        )
        for index, (start, relation, end) in enumerate(relationships):
            writer.writerow((start, relation, end, f"r:{index}", "{}"))


def test_retrieval_preserves_provenance_and_causal_status(tmp_path):
    _write_graph(tmp_path)
    results = retrieve_literature_kg_claims(
        "How can 2D perovskites improve 3D stability?", graph_root=tmp_path
    )
    assert len(results) == 1
    assert results[0].doi == "10.1000/test"
    assert results[0].causal_status == "reported_association"
    assert results[0].support_status == "evidence_linked"
    assert "treated devices" in results[0].evidence_excerpt


def test_context_states_that_directional_claims_are_not_causal_proof(tmp_path):
    _write_graph(tmp_path)
    context = build_hypothesis_kg_context(
        "2D 3D perovskite stability", graph_root=tmp_path
    )
    assert "snapshot=test-graph" in context
    assert "not causal proof" in context
    assert "DOI: 10.1000/test" in context
