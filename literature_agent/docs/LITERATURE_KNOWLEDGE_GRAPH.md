# LiteratureAgent Knowledge Graph

## Recommended Architecture

Use an **evidence-grounded claim-centered scientific property graph** as the
canonical knowledge graph.

- A DAG is useful for selected directed reasoning paths, but scientific
  literature relationships are not naturally acyclic.
- A Sankey diagram is useful for showing aggregated flows, but it is a
  visualization rather than a knowledge representation.
- A property graph supports papers, devices, materials, layers, processes,
  measurements, stability tests, scientific claims, structured observations,
  mechanisms, outcomes, and evidence as reusable nodes connected by typed
  relationships with provenance properties.

The graph separates three epistemic levels:

1. **Database facts:** strict structured fields such as device stacks, PCE, and
   stability values.
2. **Extracted observations:** material-specific optical, electrical,
   processing, and stability observations from consolidated extraction JSON.
3. **Reported scientific claims:** exact statements from paper summaries,
   explicitly represented as claims rather than universal facts.

Directional relationships such as `IMPROVES`, `REDUCES`, and `PROMOTES` retain
the source claim, paper, extraction field, and evidence links.

The runtime writer creates Neo4j-compatible bulk-import CSVs, reports, derived
views, and plots by default. Portable JSONL and GraphML are opt-in formats for
consumers that explicitly require them. Neo4j remains an optional
serving/database layer, not a requirement for building or validating the graph.

## Initial Graph Schema

Primary node labels:

```text
Paper
PaperType
Device
Material / Perovskite
Chemical
LayerMaterial
Process
Measurement / PerformanceMeasurement / StabilityMeasurement
StabilityTest
Characterization
Evidence / FigureEvidence
Dimensionality
MaterialSystem / SummaryExtractedEntity
Observation / StructuredObservation
Claim / Finding / Mechanism / Contribution / DataQualityClaim
ScientificConcept / Intervention / Outcome / MechanismConcept
ScientificProperty
```

Primary relationship types:

```text
REPORTS_DEVICE
CLASSIFIED_AS
USES_ABSORBER
HAS_A_SITE
HAS_B_SITE
HAS_X_SITE
USES_ADDITIVE
HAS_*_LAYER
NEXT_LAYER
FABRICATED_BY
HAS_PERFORMANCE_MEASUREMENT
HAS_STABILITY_TEST
HAS_STABILITY_MEASUREMENT
HAS_FIGURE_EVIDENCE
USES_CHARACTERIZATION
EVIDENCE_FOR
HAS_TEXT_EVIDENCE
HAS_SUMMARY_EVIDENCE
STUDIES_MATERIAL_SYSTEM
REPORTS_SUMMARY_PERFORMANCE
REPORTS_STRUCTURED_OBSERVATION
OBSERVES_PROPERTY
OBSERVED_FOR
ASSERTS_CLAIM
SUPPORTS_CLAIM
HAS_SUBJECT
HAS_OBJECT
IMPROVES
REDUCES
INCREASES
PROMOTES
INHIBITS
ASSOCIATED_WITH
ATTRIBUTED_TO
```

Every relationship includes at least:

```text
source_field
source_record_id
provenance_source
```

## Build Command

```powershell
cd "<repo_root>"

python .\scripts\build_literature_knowledge_graph.py `
  --records "<repo_root>\lit_outputs\csv\all_records.csv" `
  --ontology "<repo_root>\config\perovskite_ontology_library_v19.json" `
  --work-dir "<repo_root>\lit_outputs" `
  --out "<repo_root>\artifacts\literature_knowledge_graph"
```

Portable exports are optional. Request them only for a specific external
consumer by adding `--portable-formats jsonl graphml`.

The same build can be run through the main LiteratureAgent controller:

```powershell
python "<repo_root>\literature_agent_full_end_to_end_v21_3_english_sanitizer.py" `
  --pipeline_stage knowledge_graph `
  --base_csv "<repo_root>\data\Perovskite_database_content_all_data.csv" `
  --ontology_path "<repo_root>\config\perovskite_ontology_library_v19.json" `
  --work_dir "<repo_root>\lit_outputs" `
  --knowledge_graph_out_dir "<repo_root>\artifacts\literature_knowledge_graph"
```

One-record test:

```powershell
python .\scripts\build_literature_knowledge_graph.py `
  --records "<repo_root>\lit_outputs\csv\all_records.csv" `
  --ontology ".\config\perovskite_ontology_library_v19.json" `
  --work-dir "<repo_root>\lit_outputs" `
  --out ".\knowledge_graph_test" `
  --max-records 1
```

## Outputs

```text
README.txt
neo4j_import/
    nodes.csv
    relationships.csv
reports/
    graph_summary.json
    node_counts_by_label.csv
    relationship_counts_by_type.csv
    ontology_term_usage.csv
    dangling_relationships.csv
    claim_catalog.csv
    claim_counts_by_type.csv
    claim_support_status.csv
    claims_without_linked_evidence.csv
derived_views/
    sankey_schema_flows.csv
    material_process_performance_dag_edges.csv
    learned_scientific_relationships.csv
    cross_paper_relationship_patterns.csv
    paper_knowledge_cards.jsonl
plots/
    node_coverage_by_type.png
    relationship_coverage_by_type.png
    scientific_claim_coverage_by_type.png
    claim_evidence_linkage.png
    learned_scientific_relationships.png
```

The `portable/` directory is not created by default. When
`--portable-formats jsonl`, `graphml`, or both are supplied, it contains
`graph.jsonl` and/or `literature_knowledge_graph.graphml`.

## POLARIS Integration

For POLARIS, treat the graph builder as a separate resumable stage after
`extract_batch` or `vision_pass`. Use the Neo4j CSVs when a persistent graph
database is available. Request portable JSONL only when a specific agent tool
or interchange workflow needs a file-based graph.

Recommended agent queries include:

- Which spacer cations are associated with high PCE in 2D devices?
- Which device stacks appear in papers reporting stability evidence?
- Which characterization methods support a claimed mechanism?
- Which papers connect a material/additive to both performance and stability?
- Which extracted relationships still lack visual or textual evidence?
- What mechanisms are reported for spacer-cation or additive improvements?
- Which interventions repeatedly promote stability, charge transport, or
  reduced defect density across multiple papers?
- Which structured observations support or contradict a reported claim?
- Which claims are summary-only and should be prioritized for human review?

## Current Limitation

The graph intentionally reflects current extraction coverage. A relationship
derived from a summary is stored as a reported claim, not promoted to consensus
or treated as a validated universal fact. Cross-paper relationship counts are
provided for discovery, but human review or stronger evidence aggregation is
required before treating repeated claims as scientific consensus.
