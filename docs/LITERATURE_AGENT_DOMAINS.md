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
  "ontology_path": "../literature_agent/config/domain_packs/general_science.json"
}
```

Alternatively, set `LITERATURE_AGENT_DOMAIN_PROFILE`. The environment variable
overrides the JSON default. The configured `ontology_path` must point to the
ontology/domain pack that matches the selected default profile.

## Request-level selection

`POST /api/v1/literature/start_stage` accepts an optional `domain_profile`:

```json
{
  "stage": "extract_batch",
  "domain_profile": "general_science",
  "search_query": "solid electrolyte interface spectroscopy",
  "max_papers": 25,
  "extraction_profile": "general_literature"
}
```

If `domain_profile` is omitted, the configured backend default is used. Existing
clients therefore continue to run `halide_pv` with
`pce_stability_modeling` and strict family gating unless the backend default is
changed.

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
assets. Health output reports both the configured `domain_profile` and the list
of supported profile identifiers.
