# LiteratureAgent domain profiles

POLARIS can launch LiteratureAgent jobs for three structured-extraction
profiles:

- `halide_pv` (default): halide-perovskite photovoltaic extraction, database
  integration, and PCE/stability modeling.
- `oxide_perovskite`: oxide composition, structure, processing, defect, and
  property extraction.
- `general_science`: domain-neutral systems, methods, conditions,
  measurements, findings, claims, evidence, and limitations.

Set the backend default in `config/literature_agent.json`:

```json
{
  "domain_profile": "general_science",
  "ontology_path": "../literature_agent/config/domain_packs/general_science.json",
  "knowledge_graph_builder": "../literature_agent/scripts/build_literature_knowledge_graph.py"
}
```

Alternatively, set `LITERATURE_AGENT_DOMAIN_PROFILE`. The environment variable
overrides the JSON default. The configured `ontology_path` must point to the
ontology/domain pack that matches the selected default profile.
`LITERATURE_AGENT_KG_BUILDER` can override the bundled knowledge-graph builder,
but normal standalone bundles should leave it unset so the portable config path
is used.

## Request-level selection

`POST /api/v1/literature/start_stage` accepts optional `domain_profile` and
`ontology_path` values:

```json
{
  "stage": "extract_batch",
  "domain_profile": "general_science",
  "ontology_path": "literature_agent/config/domain_packs/general_science.json",
  "search_query": "solid electrolyte interface spectroscopy",
  "max_papers": 25,
  "extraction_profile": "general_literature"
}
```

If `domain_profile` is omitted, the configured backend default is used. Existing
clients therefore continue to run `halide_pv` with
`pce_stability_modeling` and strict family gating unless the backend default is
changed.

An omitted `ontology_path` uses the backend configuration. A request override
may be absolute or relative to the POLARIS backend root, but it must resolve to
an existing, readable, valid JSON file inside that backend directory. Paths
outside the backend are rejected before a subprocess starts. This keeps custom
domain packs deployable with the bundle without allowing the API to read
arbitrary host files.

For `general_science` and `oxide_perovskite`, POLARIS always sends
`--family_gating off`; a caller cannot accidentally apply the photovoltaic
family gate to another material system. If `extraction_profile` is omitted for
a non-halide request, the API uses `general_literature`.

## PCE/model safety boundary

The `pce_stability_modeling` extraction profile and the
`integrate_and_model` stage are accepted only for `halide_pv`. POLARIS rejects
those combinations before starting a subprocess for any other profile. General
and oxide campaigns can still run extraction, vision, summary sanitization,
evidence-packet, and knowledge-graph stages.

Use a separate `work_dir`, integration directory, and knowledge-graph directory
for each domain. Request-level selection changes the controller contract for one
job; it does not relocate or erase existing artifacts.

The configured controller must include the matching LiteratureAgent domain-pack
assets. The bundled generic knowledge-graph builder creates domain-neutral
`StudiedSystem`, `DomainEntity`, measurement, attribute, and evidence nodes for
non-halide records. Unrecognized `Domain_*` columns are retained as
`DomainAttribute` nodes, so a custom structured-output contract does not silently
disappear from the graph. Domain-specific relations can still be added to
`add_domain_record` when a generic attribute edge is not expressive enough.

Health output reports the configured `domain_profile`, supported profiles, and
resolved controller, ontology, and knowledge-graph-builder paths/checks.
