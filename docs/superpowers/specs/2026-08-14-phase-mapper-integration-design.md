# Phase-Mapper Integration Design

## Purpose

Integrate the "Optical-to-XRD Phase Mapper" reference application (delivered as
`POLARIS_Integration_Handoff_20260810`) into POLARIS. The reference app is a
2,874-line Streamlit dashboard plus an 8,000-line `src/` library that predicts
XRD-derived phase labels and homogeneity from PL/absorbance/composition data
for combinatorial perovskite libraries, and recommends the next wells to
measure by XRD.

The full handoff also includes a post-library spin-coating Monte Carlo
optimizer and a bundled external XRD peak-fitting GUI. Those are large,
mostly-independent subsystems and are **out of scope for this spec** — they
get their own design docs once this sub-project lands. This spec covers only
the drop-cast combinatorial-library workflow: everything in the reference
app's tab bar except "Post-library Spin Optimization".

Source of truth for behavior: the reference app itself, run locally
(`streamlit run app.py` against
`POLARIS_Integration_Handoff_20260810/`), not just its source files, per the
handoff doc's instruction. Screens verified during design: Reads, Heatmaps,
Predictions, and the spin-optimization lock gate (confirming tab structure
and the library-lock boundary described in the docs).

## Non-goals

- Spin-coating Monte Carlo optimizer (separate sub-project, depends on this one).
- The bundled external XRD peak-fitting GUI app (`external/XRD fitting/`) — decision deferred.
- Any change to the scientific/modeling logic in `src/*.py`. The handoff doc is explicit that scientific logic changes should be discussed before substituting a POLARIS-native implementation; this integration ports behavior, not algorithms.
- Full mobile parity for dense visualizations (heatmaps, spectra) — mobile v1 is a reduced view.

## Architecture

### Backend (`backend-api`)

A new `app/services/phase_mapper/` package ports the reference app's `src/`
modules with their algorithmic content unchanged:

- `pipeline.py`, `modeling.py`, `composition_features.py`, `spectral_features.py`, `plate_reader.py`, `xrd_processing.py`, `xrd_labels.py`, `xrd_auto_assignment.py`, `xrd_peak_fitting.py`, `active_learning.py`, `active_learning_review.py`, `experiment_profiles.py`, `config.py`, `utils.py`.

Only I/O changes: these modules currently read/write a single global
`outputs/` and `data/` tree (`config.yaml`-driven); the ported versions take
explicit per-user, per-library paths instead.

Around that ported core, following the existing `curve_fitting_*` /
`literature_agent_*` service conventions:

- `app/services/phase_mapper_uploads.py` — persists plate-reader, composition, and XRD uploads per user + library, mirroring `curve_fitting_uploads.py`.
- `app/services/phase_mapper_service.py` — library CRUD, run orchestration, and JSON serialization of reads/heatmaps/predictions/recommendations/spectra/peak-fits, mirroring `curve_fitting_service.py`.
- `app/services/phase_mapper_jobs.py` — runs a pipeline execution as a background subprocess using the existing job-runner pattern (`app/tools/literature_agent_job_runner.py`: a job_id, a command, a `job_dir`, and a polled `.returncode` file).
- `app/api/v1/phase_mapper.py` — REST routes (see API surface below).
- New Alembic migration adding `phase_mapper_libraries` and `phase_mapper_runs` tables for fast per-user listing. Run artifacts themselves (uploaded files, parsed CSVs, run history) stay on disk under the existing volume-backed user-data directory (`app/tools/paths.py: get_user_data_dir()`), the same as curve-fitting uploads today.

Storage layout (per user):

```
<user_data_dir>/phase_mapper/<user_id>/<library_slug>/
  inputs/                     # composition map, saved config, reference files
  uploads/plate_reader/...
  uploads/raw_xrd/...
  xrd_labels/reviewed/...     # manual review overrides, authoritative
  runs/<timestamp>/           # frozen run history (immutable after write)
  runs/latest -> runs/<timestamp>   # pointer to latest completed run
  run_index.json
```

This mirrors the reference app's own `outputs/dashboard_run_history/` /
`outputs/dashboard_run/<library>/` model, just namespaced under a user.

### API surface (new, under `/api/v1/phase-mapper`)

- `GET/POST /libraries`, `GET/PATCH/DELETE /libraries/{id}`
- `POST /libraries/{id}/uploads` (composition CSV, plate-reader workbook, raw XRD, reviewed XRD labels)
- `POST /libraries/{id}/runs` → starts a background job, returns `job_id`
- `GET /libraries/{id}/jobs/{job_id}` → job status (pending/running/succeeded/failed) + captured stderr on failure
- `GET /libraries/{id}/runs` (history list), `GET /runs/{run_id}` (detail/summary), `DELETE /runs/{run_id}`
- Per-run, per-tab data: `GET /runs/{run_id}/reads`, `/heatmaps`, `/predictions`, `/xrd-recommendations`, `/remeasurement`, `/optical-spectra`, `/xrd-spectra`, `/xrd-peak-fits`
- `POST /runs/{run_id}/xrd-labels/review` (manual label overrides)
- `GET /runs/{run_id}/download` (zip, matching the reference app's per-run ZIP export)

Contract lives in `shared-types/openapi/polaris-v1.yaml`; `shared-types/src/generated/api.ts` is regenerated from it, same as the rest of the API.

### Frontend web (`web-frontend`)

New top-level nav entry "Phase Mapper" (not nested under Agents — this is a
deterministic ML pipeline with its own multi-screen workflow, not a
conversational agent) at `app/(app)/phase-mapper/`, with a library
selector and the same tab set as the reference app: Reads, Heatmaps,
Predictions, XRD Recommendations, Remeasurement, Optical Spectra, XRD
Spectra, XRD Peak Fits, Library Runs, Downloads.

Heatmaps, spectra, and model-performance charts are rendered as native,
themed web components from JSON data (per the dataviz skill) rather than the
reference app's server-rendered matplotlib PNGs — this matches how the rest
of POLARIS's UI works, themes correctly in light/dark, and is usable on
mobile. This is more upfront work than passing through images, and each
chart's numeric output should be spot-checked against the reference app's
matplotlib rendering during acceptance testing.

### Mobile (`mobile-development`)

Reduced-view v1: library list, run status/history, predictions summary, XRD
label review actions, upload triggers. Heatmap/spectra visualization is
deferred to a later pass once the web version is validated against the
reference app.

## Data flow

1. Create/open a library → select a composition profile or upload a composition CSV → upload plate-reader workbook(s), optionally raw XRD scans and/or reviewed XRD labels. Persisted to the library's upload dir and recorded in the DB.
2. "Run Pipeline" → `POST /runs` creates a job row and starts the background subprocess against the library's persisted inputs → returns `job_id`.
3. Frontend polls job status. On completion, outputs land in a new timestamped run dir (frozen) plus an updated `latest` pointer, indexed in the DB.
4. Frontend fetches per-tab JSON scoped to a run (default `latest`) and renders it.
5. XRD label review: uploaded/edited reviewed labels are stored as overrides and take precedence over automatic proposals in the next run — manual labels remain authoritative, per the integration boundary in the handoff doc.
6. History: list frozen runs, inspect/download any run's outputs; deleting a run repairs the run index and the `latest` pointer, matching the reference app's behavior.

## Integration boundaries carried over from the handoff doc

- `library::well` remains the sample identity so wells from different libraries never merge accidentally, even when pooling prior libraries for training.
- Saved uploads, latest outputs, timestamped frozen run history, and reviewed XRD labels are all preserved — history runs are immutable after training (review/edit tools don't write back into history folders).
- Manual XRD labels are authoritative over automatic proposals.
- The finalized composition, clean-3D probability, and multitask-model uncertainty are the only things passed to the (future) downstream spin optimizer — this sub-project doesn't build that consumer, but its outputs must be shaped to support it later without rework.

## Error handling

- Upload validation keeps the reference app's warn-vs-fail semantics: fail only on a missing `well` column; warn (don't block) on incomplete well coverage or composition fractions that don't sum to ~1.
- Job failures are captured via the subprocess runner's return code and stderr, exposed through the job-status endpoint, and surfaced in the UI as a failed run (kept on disk for debugging, not marked frozen/complete).
- Same-user concurrent "Run Pipeline" requests on one library are serialized: a second request while a job is in flight returns 409 rather than racing writes into the same run directory.

## Testing

- Port the reference app's existing pytest suite (entropy, composition descriptors, synthetic PL peak extraction, synthetic absorbance edge detection, active-learning ranking, read-assignment parsing) into `backend-api/tests/phase_mapper/` largely unchanged, since the underlying logic isn't changing.
- Add API-level tests covering upload → run → poll → fetch for each tab's endpoint, against a small fixture library.
- Manual acceptance pass against the "Optical-to-XRD library workflow" checklist in `POLARIS_INTEGRATION_HANDOFF.md`, using the running reference app as the comparison baseline, and recording any behavioral differences per that doc's "Acceptance notes" format.

## Open questions for later sub-projects

- Whether the external XRD peak-fitting GUI (`external/XRD fitting/`) gets ported as its own screen, replaced by the JSON-based XRD Peak Fits tab above, or left out of POLARIS entirely.
- Spin-coating optimizer design (sub-project 2), which consumes this sub-project's finalized-composition and uncertainty outputs.
