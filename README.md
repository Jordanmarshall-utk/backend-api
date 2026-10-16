# backend-api

POLARIS FastAPI backend — agents, watcher, MCP orchestrator, and persistence.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python init_db.py
uvicorn app.main:app --reload --port 8080
```

Health: `GET http://localhost:8080/api/v1/health`

LiteratureAgent supports configurable scientific domains while retaining
`halide_pv` as the default. See
**[docs/LITERATURE_AGENT_DOMAINS.md](./docs/LITERATURE_AGENT_DOMAINS.md)** for
configuration, API examples, and the PCE-model safety boundary.

### Deploy to Railway (production API)

See **[docs/DEPLOY-RAILWAY.md](./docs/DEPLOY-RAILWAY.md)** for volume setup, env vars, and pointing Vercel `NEXT_PUBLIC_API_URL` at your Railway domain.

### Run API + web + mobile together

From [web-frontend](../web-frontend) (with this repo’s `.venv` ready):

```bash
npm run dev:stack
```

## LLM providers

- **Qwen** — Hugging Face router (`HF_TOKEN` / settings API key)
- **Google Gemini** — `GEMINI_API_KEY` or `GOOGLE_API_KEY`, or save via `PATCH /api/v1/settings`

List models: `GET /api/v1/llm/providers`

### Hypothesis agent performance

The flow is **inherently multi-step** (clarify → explore → refine → synthesize). Each step is a separate LLM round-trip, so latency adds up—especially on Gemini free tier with `GEMINI_MIN_INTERVAL_SEC=6` (default), which spaces calls to avoid 429s.

**Defaults (API / headless)** reduce calls vs the original Streamlit path:

| Setting | Default | Effect |
|--------|---------|--------|
| `HYPOTHESIS_FAST_MODE` | `1` | After clarify, one combined call for Socratic pass + 3 thoughts (~2 calls on submit instead of ~4) |
| `HYPOTHESIS_SKIP_READINESS_CHECK` | `1` | No extra LLM call before each next-step pick |
| `HYPOTHESIS_SKIP_SOCRATIC_ANSWERS` | off | Set `1` to skip the self-answer step (saves another call if not using fast mode) |
| `HYPOTHESIS_SKIP_ANALYSIS_ON_GENERATE` | off | Set `1` to return hypothesis only (skips analysis rubric LLM) |
| `HYPOTHESIS_CONTEXT_MAX_CHARS` | `4500` | Truncates long prompts for faster/cheaper requests |

**Provider tips:** Prefer **`gemini-2.0-flash-lite`** or **Qwen** on a paid/low-latency tier; lower `GEMINI_MIN_INTERVAL_SEC` only if your quota allows (risk of 429). Wait ~60s and retry after 429, or enable billing for higher quotas.

## Layout

- `app/api/v1/` — REST routes (OpenAPI contract in `@polaris/shared-types`)
- `app/agents/`, `app/tools/`, `app/watcher/` — ported from `polaris_ahmadi`
- `app/core/` — config, auth, session DTOs
- `migrations/` — Alembic (Postgres / Supabase)
- `tests/` — pytest

## Auth

Development defaults to `AUTH_DISABLED=true`. Production: set Supabase JWT secret and disable auth bypass.

## Migrated from

[CyberA183/polaris_ahmadi](https://github.com/CyberA183/polaris_ahmadi) — see [MIGRATION.md](./MIGRATION.md).
