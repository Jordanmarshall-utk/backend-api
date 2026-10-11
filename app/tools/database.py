"""
SQLAlchemy async DatabaseManager — supports both SQLite (aiosqlite) and Postgres (asyncpg).
Public API is identical to the original sqlite3 implementation.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.engine import get_async_engine

# ── keep all the constants unchanged ──────────────────────────────────────────
EXPERIMENT_SCOPED_KEYS = frozenset([
    "stage", "last_hypothesis", "experimental_outputs", "analysis_results", "curve_fitting_results",
    "interactions", "hypothesis_ready", "stop_hypothesis", "hypothesis_round_count",
    "workflow_active", "workflow_step", "workflow_completed", "workflow_experiment_started",
    "workflow_experiment_completed", "workflow_experiment_outputs", "research_goal",
    "current_workflow", "workflow_steps", "conversation_history", "allow_followup",
    "last_reasoning_plan",
    "selected_wells", "plate_format", "last_processed_trigger_time", "watcher_auto_triggered_file",
    "watcher_auto_trigger_time", "auto_triggered_results", "auto_triggered_results_file",
    "auto_run_curve_fitting", "auto_run_data_file", "auto_run_comp_file", "auto_run_params",
    "auto_ml_after_curve_fitting", "auto_route_to_analysis", "workflow_curve_fitting_files",
    "ml_auto_json_path", "ml_auto_csv_path", "ml_auto_composition_path",
    "manual_clarified_question", "manual_socratic_questions", "manual_socratic_answers",
    "manual_thoughts", "manual_hypothesis", "optimization_model_choice", "ml_model_config",
    "gp_results", "gp_results_available", "analysis_ready", "analysis_recommendations",
    "analysis_hypothesis_fingerprint",
    "gp_suggested_compositions", "analysis_full_report", "analysis_feedback", "next_agent",
    "pending_additional_question", "workflow_ml_model_choice", "workflow_curve_fitting_completed",
    "workflow_created_at", "demo_workflow_running", "prompt_session", "current_prompt_session_id",
])

DEFAULT_APP_CONFIG = {
    "start_time": None,
    "api_key": "",
    "api_key_source": "",
    "current_user_id": "",
    "current_experiment_id": 0,
    "llm_provider": "qwen",
    "llm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen_base_url": "https://router.huggingface.co/v1",
    "stage": "initial",
    "cf_data_path": "",
    "cf_comp_path": "",
    "experimental_mode": False,
    "routing_mode": "Autonomous (LLM)",
    "reasoning_mode": "Shadow",
    "max_hypothesis_rounds": 5,
    "experiment_memory_file": "experiment_memory.json",
    "experiment_data_dir": "data",
    "watcher_directory": "",
    "watcher_results_dir": "results",
    "watcher_enabled": False,
    "watcher_port": 8000,
}

DEFAULT_EXPERIMENTAL_CONSTRAINTS = {
    "techniques": [], "equipment": [], "parameters": [], "focus_areas": [],
    "liquid_handling": {"max_volume_per_mixture": 50, "instruments": [], "plate_format": "96-well", "materials": [], "csv_path": "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"},
}

DEFAULT_JUPYTER_CONFIG = {
    "server_url": "http://10.140.141.160:48888/", "token": "", "upload_enabled": False, "notebook_path": "Automated Agent",
}

DEFAULT_SESSION_STATE = {
    "conversation_events": [], "current_prompt_session_id": None,
    "manual_workflow": ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"],
    "workflow_index": 0, "workflow_auto_flags": {}, "current_workflow_name": None,
    "uploaded_files": [], "hypothesis_ready": False, "last_hypothesis": None,
    "experimental_outputs": None, "stop_hypothesis": False, "hypothesis_round_count": 0,
    "last_reasoning_plan": None,
    "workflow_active": False, "workflow_step": "idle", "workflow_completed": False,
    "workflow_experiment_started": False, "workflow_experiment_completed": False,
    "workflow_experiment_outputs": None, "research_goal": "",
}

DEFAULT_AGENT_COUNTS = {
    "hypothesis": 0, "experiment": 0, "curve_fit": 0, "analysis": 0, "router": 0, "watcher": 0,
}


def _run(coro):
    """Run an async coroutine synchronously.

    When called from within a running event loop (e.g. FastAPI request), each
    call opens a new connection via ThreadPoolExecutor + asyncio.run(), bypassing
    the engine pool. For high-throughput paths, prefer async methods directly.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an async context (e.g. FastAPI request) — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _is_postgres(engine: AsyncEngine) -> bool:
    return str(engine.url).startswith("postgresql")


def _upsert_sql(table: str, pk_col: str, pk_val, columns: dict, engine: AsyncEngine) -> tuple:
    """Build upsert SQL compatible with both SQLite and Postgres."""
    cols = list(columns.keys())
    all_cols = [pk_col] + cols
    all_vals = [pk_val] + [columns[c] for c in cols]
    placeholders = ", ".join(f":{c}" for c in all_cols)
    params = dict(zip(all_cols, all_vals))
    params[pk_col] = pk_val

    if _is_postgres(engine):
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
        sql = (
            f"INSERT INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_set}"
        )
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders})"
    return sql, params


class DatabaseManager:
    """SQLAlchemy async persistence — SQLite (aiosqlite) or Postgres (asyncpg)."""

    def __init__(self, engine: Optional[AsyncEngine] = None):
        self._engine = engine or get_async_engine()

    # ── schema ────────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        _run(self._init_schema_async())

    async def _init_schema_async(self) -> None:
        pg = _is_postgres(self._engine)
        serial = "BIGSERIAL" if pg else "INTEGER"
        autoincrement = "AUTOINCREMENT" if not pg else ""
        pk1_col = "id BIGINT PRIMARY KEY DEFAULT 1 CHECK (id = 1)" if pg else "id INTEGER PRIMARY KEY CHECK (id = 1)"

        ddl_statements = [
            """CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value_text TEXT,
                value_int BIGINT,
                value_real DOUBLE PRECISION,
                value_json TEXT
            )""",
            f"""CREATE TABLE IF NOT EXISTS experimental_constraints (
                {pk1_col},
                techniques_json TEXT,
                equipment_json TEXT,
                parameters_json TEXT,
                focus_areas_json TEXT,
                liquid_handling_json TEXT
            )""",
            f"""CREATE TABLE IF NOT EXISTS jupyter_config (
                {pk1_col},
                server_url TEXT,
                token TEXT,
                upload_enabled INTEGER,
                notebook_path TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            f"""CREATE TABLE IF NOT EXISTS experiments (
                id {serial} PRIMARY KEY {autoincrement},
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""",
            """CREATE TABLE IF NOT EXISTS experiment_data (
                experiment_id BIGINT PRIMARY KEY,
                state_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )""",
            f"""CREATE TABLE IF NOT EXISTS conversation_events (
                id {serial} PRIMARY KEY {autoincrement},
                experiment_id BIGINT,
                type TEXT NOT NULL,
                mode TEXT NOT NULL,
                prompt_session_id TEXT,
                timestamp TEXT NOT NULL,
                payload_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )""",
            """CREATE TABLE IF NOT EXISTS agent_usage_counts (
                agent_name TEXT PRIMARY KEY,
                count BIGINT NOT NULL DEFAULT 0
            )""",
            f"""CREATE TABLE IF NOT EXISTS workflows (
                id {serial} PRIMARY KEY {autoincrement},
                name TEXT UNIQUE NOT NULL,
                created_at TEXT,
                ml_model_choice TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS workflow_steps (
                workflow_id BIGINT NOT NULL,
                step_order BIGINT NOT NULL,
                name TEXT NOT NULL,
                automatic INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (workflow_id, step_order),
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )""",
            f"""CREATE TABLE IF NOT EXISTS uploaded_files (
                id {serial} PRIMARY KEY {autoincrement},
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )""",
            f"""CREATE TABLE IF NOT EXISTS session_state (
                {pk1_col},
                state_json TEXT
            )""",
            f"""CREATE TABLE IF NOT EXISTS negative_hypotheses (
                id {serial} PRIMARY KEY {autoincrement},
                hypothesis_text TEXT NOT NULL,
                status TEXT NOT NULL,
                research_question TEXT,
                analysis_summary TEXT,
                context_json TEXT,
                created_at TEXT NOT NULL
            )""",
            f"""CREATE TABLE IF NOT EXISTS hypothesis_outcomes (
                id {serial} PRIMARY KEY {autoincrement},
                hypothesis_text TEXT NOT NULL,
                material_hint TEXT,
                status TEXT NOT NULL,
                evidence_summary TEXT,
                source TEXT,
                created_at TEXT NOT NULL
            )""",
        ]

        async with self._engine.begin() as conn:
            for stmt in ddl_statements:
                await conn.execute(text(stmt))

    def ensure_defaults(self) -> None:
        _run(self._ensure_defaults_async())

    async def _ensure_defaults_async(self) -> None:
        await self._init_schema_async()
        async with self._engine.begin() as conn:
            # app_config defaults
            for key, value in DEFAULT_APP_CONFIG.items():
                if value is None:
                    continue
                existing = (await conn.execute(
                    text("SELECT 1 FROM app_config WHERE key = :k"), {"k": key}
                )).fetchone()
                if existing is None:
                    await self._set_config_value_async(conn, key, value)

            # experimental_constraints
            if not (await conn.execute(text("SELECT 1 FROM experimental_constraints WHERE id = 1"))).fetchone():
                ec = DEFAULT_EXPERIMENTAL_CONSTRAINTS
                await conn.execute(text(
                    "INSERT INTO experimental_constraints (id, techniques_json, equipment_json, parameters_json, focus_areas_json, liquid_handling_json) "
                    "VALUES (1, :t, :e, :p, :f, :l)"
                ), {"t": json.dumps(ec["techniques"]), "e": json.dumps(ec["equipment"]),
                    "p": json.dumps(ec["parameters"]), "f": json.dumps(ec["focus_areas"]),
                    "l": json.dumps(ec["liquid_handling"])})

            # jupyter_config
            if not (await conn.execute(text("SELECT 1 FROM jupyter_config WHERE id = 1"))).fetchone():
                jc = DEFAULT_JUPYTER_CONFIG
                await conn.execute(text(
                    "INSERT INTO jupyter_config (id, server_url, token, upload_enabled, notebook_path) VALUES (1, :s, :t, :u, :n)"
                ), {"s": jc["server_url"], "t": jc["token"], "u": int(jc["upload_enabled"]), "n": jc["notebook_path"]})

            # agent_usage_counts
            for agent, count in DEFAULT_AGENT_COUNTS.items():
                if not (await conn.execute(text("SELECT 1 FROM agent_usage_counts WHERE agent_name = :a"), {"a": agent})).fetchone():
                    await conn.execute(text("INSERT INTO agent_usage_counts (agent_name, count) VALUES (:a, :c)"), {"a": agent, "c": count})

            # session_state
            if not (await conn.execute(text("SELECT 1 FROM session_state WHERE id = 1"))).fetchone():
                state = dict(DEFAULT_SESSION_STATE)
                state["current_prompt_session_id"] = str(uuid.uuid4())
                state["start_time"] = datetime.now().timestamp()
                await conn.execute(text("INSERT INTO session_state (id, state_json) VALUES (1, :s)"), {"s": json.dumps(state)})

    async def _set_config_value_async(self, conn, key: str, value: Any) -> None:
        pg = _is_postgres(self._engine)
        upsert = "INSERT INTO app_config (key, {col}) VALUES (:k, :v) ON CONFLICT (key) DO UPDATE SET {col} = EXCLUDED.{col}" if pg else "INSERT OR REPLACE INTO app_config (key, {col}) VALUES (:k, :v)"
        if isinstance(value, bool):
            await conn.execute(text(upsert.format(col="value_int")), {"k": key, "v": 1 if value else 0})
        elif isinstance(value, int):
            await conn.execute(text(upsert.format(col="value_int")), {"k": key, "v": value})
        elif isinstance(value, float):
            await conn.execute(text(upsert.format(col="value_real")), {"k": key, "v": value})
        elif isinstance(value, (dict, list)):
            await conn.execute(text(upsert.format(col="value_json")), {"k": key, "v": json.dumps(value)})
        else:
            await conn.execute(text(upsert.format(col="value_text")), {"k": key, "v": str(value) if value is not None else ""})

    # ── get ───────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return _run(self._get_async(key, default))

    async def _get_async(self, key: str, default: Any = None) -> Any:
        async with self._engine.connect() as conn:
            if key in DEFAULT_APP_CONFIG:
                row = (await conn.execute(
                    text("SELECT value_text, value_int, value_real, value_json FROM app_config WHERE key = :k"),
                    {"k": key}
                )).fetchone()
                if row is None:
                    return DEFAULT_APP_CONFIG.get(key, default)
                for i, col in enumerate(["value_text", "value_int", "value_real", "value_json"]):
                    v = row[i]
                    if v is not None:
                        if col == "value_json":
                            try:
                                return json.loads(v)
                            except json.JSONDecodeError:
                                return v
                        if col == "value_int" and key == "experimental_mode":
                            return bool(v)
                        return v
                return default

            if key == "experimental_constraints":
                row = (await conn.execute(text(
                    "SELECT techniques_json, equipment_json, parameters_json, focus_areas_json, liquid_handling_json FROM experimental_constraints WHERE id = 1"
                ))).fetchone()
                if row is None:
                    return DEFAULT_EXPERIMENTAL_CONSTRAINTS
                return {
                    "techniques": json.loads(row[0] or "[]"),
                    "equipment": json.loads(row[1] or "[]"),
                    "parameters": json.loads(row[2] or "[]"),
                    "focus_areas": json.loads(row[3] or "[]"),
                    "liquid_handling": json.loads(row[4] or "{}"),
                }

            if key == "jupyter_config":
                row = (await conn.execute(text(
                    "SELECT server_url, token, upload_enabled, notebook_path FROM jupyter_config WHERE id = 1"
                ))).fetchone()
                if row is None:
                    return DEFAULT_JUPYTER_CONFIG
                return {"server_url": row[0] or "", "token": row[1] or "", "upload_enabled": bool(row[2]) if row[2] is not None else False, "notebook_path": row[3] or "Automated Agent"}

            if key == "workflows":
                return await self._get_workflows_async(conn)

            if key == "conversation_events":
                return await self._get_conversation_events_async(conn)

            if key == "uploaded_files":
                return await self._get_uploaded_files_async(conn)

            if key == "agent_usage_counts":
                rows = (await conn.execute(text("SELECT agent_name, count FROM agent_usage_counts"))).fetchall()
                return {r[0]: r[1] for r in rows} if rows else dict(DEFAULT_AGENT_COUNTS)

            if key in EXPERIMENT_SCOPED_KEYS:
                exp_id_row = (await conn.execute(
                    text("SELECT value_int FROM app_config WHERE key = 'current_experiment_id'")
                )).fetchone()
                exp_id = int(exp_id_row[0]) if exp_id_row and exp_id_row[0] else 0
                if exp_id > 0:
                    row = (await conn.execute(
                        text("SELECT state_json FROM experiment_data WHERE experiment_id = :id"),
                        {"id": exp_id}
                    )).fetchone()
                    if row:
                        state = json.loads(row[0] or "{}")
                        if key in state:
                            val = state[key]
                            return set(val) if key == "selected_wells" and isinstance(val, list) else val
                row = (await conn.execute(text("SELECT state_json FROM session_state WHERE id = 1"))).fetchone()
                if row is None:
                    return default
                state = json.loads(row[0] or "{}")
                if key in state:
                    val = state[key]
                    return set(val) if key == "selected_wells" and isinstance(val, list) else val
                return default

            return default

    # ── set ───────────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        _run(self._set_async(key, value))

    async def _set_async(self, key: str, value: Any) -> None:
        pg = _is_postgres(self._engine)
        async with self._engine.begin() as conn:
            if key in DEFAULT_APP_CONFIG:
                await self._set_config_value_async(conn, key, value)
                return

            if key == "experimental_constraints":
                ec = value
                upsert = (
                    "INSERT INTO experimental_constraints (id, techniques_json, equipment_json, parameters_json, focus_areas_json, liquid_handling_json) "
                    "VALUES (1, :t, :e, :p, :f, :l) ON CONFLICT (id) DO UPDATE SET "
                    "techniques_json=EXCLUDED.techniques_json, equipment_json=EXCLUDED.equipment_json, "
                    "parameters_json=EXCLUDED.parameters_json, focus_areas_json=EXCLUDED.focus_areas_json, "
                    "liquid_handling_json=EXCLUDED.liquid_handling_json"
                ) if pg else (
                    "INSERT OR REPLACE INTO experimental_constraints (id, techniques_json, equipment_json, parameters_json, focus_areas_json, liquid_handling_json) "
                    "VALUES (1, :t, :e, :p, :f, :l)"
                )
                await conn.execute(text(upsert), {
                    "t": json.dumps(ec.get("techniques", [])), "e": json.dumps(ec.get("equipment", [])),
                    "p": json.dumps(ec.get("parameters", [])), "f": json.dumps(ec.get("focus_areas", [])),
                    "l": json.dumps(ec.get("liquid_handling", {}))
                })
                return

            if key == "jupyter_config":
                jc = value
                upsert = (
                    "INSERT INTO jupyter_config (id, server_url, token, upload_enabled, notebook_path) "
                    "VALUES (1, :s, :t, :u, :n) ON CONFLICT (id) DO UPDATE SET "
                    "server_url=EXCLUDED.server_url, token=EXCLUDED.token, upload_enabled=EXCLUDED.upload_enabled, notebook_path=EXCLUDED.notebook_path"
                ) if pg else (
                    "INSERT OR REPLACE INTO jupyter_config (id, server_url, token, upload_enabled, notebook_path) VALUES (1, :s, :t, :u, :n)"
                )
                await conn.execute(text(upsert), {
                    "s": jc.get("server_url", ""), "t": jc.get("token", ""),
                    "u": int(jc.get("upload_enabled", False)), "n": jc.get("notebook_path", "Automated Agent")
                })
                return

            if key == "agent_usage_counts":
                for agent, count in value.items():
                    upsert = (
                        "INSERT INTO agent_usage_counts (agent_name, count) VALUES (:a, :c) "
                        "ON CONFLICT (agent_name) DO UPDATE SET count=EXCLUDED.count"
                    ) if pg else "INSERT OR REPLACE INTO agent_usage_counts (agent_name, count) VALUES (:a, :c)"
                    await conn.execute(text(upsert), {"a": agent, "c": count})
                return

            if key == "workflows" and isinstance(value, dict):
                for wf_name, wf_data in value.items():
                    if isinstance(wf_data, dict):
                        await self._save_workflow_async(conn, wf_name, wf_data.get("steps", []), wf_data.get("ml_model_choice"), pg)
                return

            if key == "selected_wells" and isinstance(value, set):
                value = list(value)

            exp_id_row = (await conn.execute(
                text("SELECT value_int FROM app_config WHERE key = 'current_experiment_id'")
            )).fetchone()
            exp_id = int(exp_id_row[0]) if exp_id_row and exp_id_row[0] else 0

            if key in EXPERIMENT_SCOPED_KEYS and exp_id > 0:
                row = (await conn.execute(
                    text("SELECT state_json FROM experiment_data WHERE experiment_id = :id"), {"id": exp_id}
                )).fetchone()
                state = json.loads(row[0] or "{}") if row else {}
                state[key] = value
                upsert = (
                    "INSERT INTO experiment_data (experiment_id, state_json) VALUES (:id, :s) "
                    "ON CONFLICT (experiment_id) DO UPDATE SET state_json=EXCLUDED.state_json"
                ) if pg else "INSERT OR REPLACE INTO experiment_data (experiment_id, state_json) VALUES (:id, :s)"
                await conn.execute(text(upsert), {"id": exp_id, "s": json.dumps(state)})
                await conn.execute(text("UPDATE experiments SET updated_at = :ts WHERE id = :id"),
                                   {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": exp_id})
            else:
                row = (await conn.execute(text("SELECT state_json FROM session_state WHERE id = 1"))).fetchone()
                state = json.loads(row[0] or "{}") if row else {}
                state[key] = value
                await conn.execute(text("UPDATE session_state SET state_json = :s WHERE id = 1"), {"s": json.dumps(state)})

    # ── conversation events ───────────────────────────────────────────────────

    def clear_conversation_events(self, experiment_id: Optional[int] = None) -> None:
        _run(self._clear_conversation_events_async(experiment_id))

    async def _clear_conversation_events_async(self, experiment_id: Optional[int] = None) -> None:
        async with self._engine.begin() as conn:
            if experiment_id and experiment_id > 0:
                await conn.execute(text("DELETE FROM conversation_events WHERE experiment_id = :id"), {"id": experiment_id})
            else:
                await conn.execute(text("DELETE FROM conversation_events"))

    def get_conversation_events(self, experiment_id: Optional[int] = None) -> List[Dict]:
        return _run(self._get_conversation_events_with_lookup(experiment_id))

    async def _get_conversation_events_with_lookup(self, experiment_id: Optional[int] = None) -> List[Dict]:
        async with self._engine.connect() as conn:
            if experiment_id is None:
                exp_id_row = (await conn.execute(
                    text("SELECT value_int FROM app_config WHERE key = 'current_experiment_id'")
                )).fetchone()
                experiment_id = int(exp_id_row[0]) if exp_id_row and exp_id_row[0] else None
            return await self._get_conversation_events_async(conn, experiment_id)

    async def _get_conversation_events_async(self, conn, experiment_id: Optional[int] = None) -> List[Dict]:
        if experiment_id and experiment_id > 0:
            rows = (await conn.execute(
                text("SELECT type, mode, prompt_session_id, timestamp, payload_json FROM conversation_events WHERE experiment_id = :id ORDER BY id"),
                {"id": experiment_id}
            )).fetchall()
        else:
            rows = (await conn.execute(
                text("SELECT type, mode, prompt_session_id, timestamp, payload_json FROM conversation_events ORDER BY id")
            )).fetchall()
        return [{"type": r[0], "mode": r[1], "prompt_session_id": r[2], "timestamp": r[3], "payload": json.loads(r[4] or "{}")} for r in rows]

    def append_conversation_event(self, event_type: str, mode: str, prompt_session_id: str, payload: dict) -> None:
        _run(self._append_conversation_event_async(event_type, mode, prompt_session_id, payload))

    async def _append_conversation_event_async(self, event_type: str, mode: str, prompt_session_id: str, payload: dict) -> None:
        async with self._engine.begin() as conn:
            exp_id_row = (await conn.execute(
                text("SELECT value_int FROM app_config WHERE key = 'current_experiment_id'")
            )).fetchone()
            exp_id = int(exp_id_row[0]) if exp_id_row and exp_id_row[0] else None
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if exp_id and exp_id > 0:
                await conn.execute(text(
                    "INSERT INTO conversation_events (experiment_id, type, mode, prompt_session_id, timestamp, payload_json) VALUES (:eid, :t, :m, :ps, :ts, :p)"
                ), {"eid": exp_id, "t": event_type, "m": mode, "ps": prompt_session_id, "ts": ts, "p": json.dumps(payload)})
            else:
                await conn.execute(text(
                    "INSERT INTO conversation_events (type, mode, prompt_session_id, timestamp, payload_json) VALUES (:t, :m, :ps, :ts, :p)"
                ), {"t": event_type, "m": mode, "ps": prompt_session_id, "ts": ts, "p": json.dumps(payload)})

    # ── workflows ─────────────────────────────────────────────────────────────

    def get_workflows(self) -> Dict[str, Dict]:
        return _run(self._get_workflows_wrapper())

    async def _get_workflows_wrapper(self) -> Dict[str, Dict]:
        async with self._engine.connect() as conn:
            return await self._get_workflows_async(conn)

    async def _get_workflows_async(self, conn) -> Dict[str, Dict]:
        rows = (await conn.execute(text("SELECT id, name, created_at, ml_model_choice FROM workflows"))).fetchall()
        result = {}
        for r in rows:
            wf_id, name, created_at, ml_model_choice = r
            steps = (await conn.execute(
                text("SELECT name, automatic FROM workflow_steps WHERE workflow_id = :id ORDER BY step_order"),
                {"id": wf_id}
            )).fetchall()
            result[name] = {
                "name": name,
                "steps": [{"name": s[0], "automatic": bool(s[1])} for s in steps],
                "ml_model_choice": ml_model_choice,
                "created_at": created_at or "",
            }
        return result

    def save_workflow(self, name: str, steps: List[Dict], ml_model_choice: Optional[str] = None) -> None:
        _run(self._save_workflow_wrapper(name, steps, ml_model_choice))

    async def _save_workflow_wrapper(self, name: str, steps: List[Dict], ml_model_choice: Optional[str] = None) -> None:
        async with self._engine.begin() as conn:
            await self._save_workflow_async(conn, name, steps, ml_model_choice, _is_postgres(self._engine))

    async def _save_workflow_async(self, conn, name: str, steps: List[Dict], ml_model_choice: Optional[str], pg: bool) -> None:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = (await conn.execute(text("SELECT id FROM workflows WHERE name = :n"), {"n": name})).fetchone()
        if row:
            wf_id = row[0]
            await conn.execute(text("DELETE FROM workflow_steps WHERE workflow_id = :id"), {"id": wf_id})
            await conn.execute(text("UPDATE workflows SET created_at = :ca, ml_model_choice = :ml WHERE id = :id"),
                               {"ca": created_at, "ml": ml_model_choice or "", "id": wf_id})
        else:
            if pg:
                result = await conn.execute(text(
                    "INSERT INTO workflows (name, created_at, ml_model_choice) VALUES (:n, :ca, :ml) RETURNING id"
                ), {"n": name, "ca": created_at, "ml": ml_model_choice or ""})
                wf_id = result.fetchone()[0]
            else:
                await conn.execute(text(
                    "INSERT INTO workflows (name, created_at, ml_model_choice) VALUES (:n, :ca, :ml)"
                ), {"n": name, "ca": created_at, "ml": ml_model_choice or ""})
                wf_id = (await conn.execute(text("SELECT last_insert_rowid()"))).fetchone()[0]
        for i, step in enumerate(steps):
            await conn.execute(text(
                "INSERT INTO workflow_steps (workflow_id, step_order, name, automatic) VALUES (:wid, :so, :n, :a)"
            ), {"wid": wf_id, "so": i, "n": step.get("name", ""), "a": int(step.get("automatic", False))})

    def delete_workflow(self, name: str) -> None:
        _run(self._delete_workflow_async(name))

    async def _delete_workflow_async(self, name: str) -> None:
        async with self._engine.begin() as conn:
            row = (await conn.execute(text("SELECT id FROM workflows WHERE name = :n"), {"n": name})).fetchone()
            if row:
                wf_id = row[0]
                await conn.execute(text("DELETE FROM workflow_steps WHERE workflow_id = :id"), {"id": wf_id})
                await conn.execute(text("DELETE FROM workflows WHERE id = :id"), {"id": wf_id})

    def get_workflow_steps(self, workflow_id: int) -> List[Dict]:
        return _run(self._get_workflow_steps_async(workflow_id))

    async def _get_workflow_steps_async(self, workflow_id: int) -> List[Dict]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text("SELECT name, automatic FROM workflow_steps WHERE workflow_id = :id ORDER BY step_order"),
                {"id": workflow_id}
            )).fetchall()
            return [{"name": r[0], "automatic": bool(r[1])} for r in rows]

    # ── uploaded files ────────────────────────────────────────────────────────

    def get_uploaded_files(self) -> List[Dict]:
        return _run(self._get_uploaded_files_wrapper())

    async def _get_uploaded_files_wrapper(self) -> List[Dict]:
        async with self._engine.connect() as conn:
            return await self._get_uploaded_files_async(conn)

    async def _get_uploaded_files_async(self, conn) -> List[Dict]:
        rows = (await conn.execute(text("SELECT name, path, timestamp FROM uploaded_files ORDER BY id"))).fetchall()
        return [{"name": r[0], "path": r[1], "timestamp": r[2]} for r in rows]

    def add_uploaded_file(self, filename: str, path: str) -> None:
        _run(self._add_uploaded_file_async(filename, path))

    async def _add_uploaded_file_async(self, filename: str, path: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO uploaded_files (name, path, timestamp) VALUES (:n, :p, :ts)"
            ), {"n": filename, "p": path, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    # ── session state ─────────────────────────────────────────────────────────

    def clear_session_state(self, keep_keys: Optional[List[str]] = None) -> None:
        _run(self._clear_session_state_async(keep_keys or []))

    async def _clear_session_state_async(self, keep_keys: List[str]) -> None:
        async with self._engine.begin() as conn:
            row = (await conn.execute(text("SELECT state_json FROM session_state WHERE id = 1"))).fetchone()
            state = json.loads(row[0] or "{}") if row else {}
            new_state = {k: v for k, v in state.items() if k in keep_keys}
            await conn.execute(text("UPDATE session_state SET state_json = :s WHERE id = 1"), {"s": json.dumps(new_state)})
            # Reset experiment-scoped app_config key; DELETE lets get() fall back to DEFAULT_APP_CONFIG["stage"] = "initial"
            if "stage" not in keep_keys:
                await conn.execute(text("DELETE FROM app_config WHERE key = 'stage'"))

    def clear_all_except(self, keep_keys: List[str]) -> None:
        _run(self._clear_all_except_async(keep_keys))

    async def _clear_all_except_async(self, keep_keys: List[str]) -> None:
        async with self._engine.begin() as conn:
            for k in ["api_key", "api_key_source", "start_time"]:
                if k not in keep_keys:
                    await conn.execute(text("DELETE FROM app_config WHERE key = :k"), {"k": k})
            state = dict(DEFAULT_SESSION_STATE)
            state["current_prompt_session_id"] = str(uuid.uuid4())
            state["start_time"] = datetime.now().timestamp()
            await conn.execute(text("UPDATE session_state SET state_json = :s WHERE id = 1"), {"s": json.dumps(state)})

    # ── users ─────────────────────────────────────────────────────────────────

    def create_user(self, user_id: str, name: str = "") -> None:
        _run(self._create_user_async(user_id, name))

    async def _create_user_async(self, user_id: str, name: str) -> None:
        pg = _is_postgres(self._engine)
        async with self._engine.begin() as conn:
            upsert = (
                "INSERT INTO users (id, name, created_at) VALUES (:id, :n, :ca) ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name"
            ) if pg else "INSERT OR REPLACE INTO users (id, name, created_at) VALUES (:id, :n, :ca)"
            await conn.execute(text(upsert), {"id": user_id, "n": name or user_id, "ca": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    def get_user(self, user_id: str) -> Optional[Dict]:
        return _run(self._get_user_async(user_id))

    async def _get_user_async(self, user_id: str) -> Optional[Dict]:
        async with self._engine.connect() as conn:
            row = (await conn.execute(text("SELECT id, name, created_at FROM users WHERE id = :id"), {"id": user_id})).fetchone()
            return {"id": row[0], "name": row[1], "created_at": row[2]} if row else None

    def list_users(self) -> List[Dict]:
        return _run(self._list_users_async())

    async def _list_users_async(self) -> List[Dict]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text("SELECT id, name, created_at FROM users ORDER BY created_at DESC"))).fetchall()
            return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    # ── experiments ───────────────────────────────────────────────────────────

    def create_experiment(self, user_id: str, name: str = "") -> int:
        return _run(self._create_experiment_async(user_id, name))

    async def _create_experiment_async(self, user_id: str, name: str) -> int:
        pg = _is_postgres(self._engine)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = name or f"Experiment {now[:19]}"
        async with self._engine.begin() as conn:
            if pg:
                result = await conn.execute(text(
                    "INSERT INTO experiments (user_id, name, created_at, updated_at) VALUES (:uid, :n, :ca, :ua) RETURNING id"
                ), {"uid": user_id, "n": name, "ca": now, "ua": now})
                exp_id = result.fetchone()[0]
            else:
                await conn.execute(text(
                    "INSERT INTO experiments (user_id, name, created_at, updated_at) VALUES (:uid, :n, :ca, :ua)"
                ), {"uid": user_id, "n": name, "ca": now, "ua": now})
                exp_id = (await conn.execute(text("SELECT last_insert_rowid()"))).fetchone()[0]
            await conn.execute(text(
                "INSERT INTO experiment_data (experiment_id, state_json) VALUES (:id, :s)"
            ), {"id": exp_id, "s": json.dumps(dict(DEFAULT_SESSION_STATE))})
            return exp_id

    def list_experiments(self, user_id: str, limit: int = 50) -> List[Dict]:
        return _run(self._list_experiments_async(user_id, limit))

    async def _list_experiments_async(self, user_id: str, limit: int) -> List[Dict]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT id, name, created_at, updated_at FROM experiments WHERE user_id = :uid ORDER BY updated_at DESC LIMIT :lim"
            ), {"uid": user_id, "lim": limit})).fetchall()
            return [{"id": r[0], "name": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        return _run(self._get_experiment_async(experiment_id))

    async def _get_experiment_async(self, experiment_id: int) -> Optional[Dict]:
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT id, user_id, name, created_at, updated_at FROM experiments WHERE id = :id"
            ), {"id": experiment_id})).fetchone()
            return {"id": row[0], "user_id": row[1], "name": row[2], "created_at": row[3], "updated_at": row[4]} if row else None

    def set_current_experiment(self, experiment_id: int, user_id: str = "") -> None:
        _run(self._set_current_experiment_async(experiment_id, user_id))

    async def _set_current_experiment_async(self, experiment_id: int, user_id: str) -> None:
        async with self._engine.begin() as conn:
            if user_id:
                await self._set_config_value_async(conn, "current_user_id", user_id)
            await self._set_config_value_async(conn, "current_experiment_id", experiment_id)

    def load_experiment_into_session(self, experiment_id: int) -> None:
        _run(self._load_experiment_into_session_async(experiment_id))

    async def _load_experiment_into_session_async(self, experiment_id: int) -> None:
        async with self._engine.begin() as conn:
            row = (await conn.execute(text(
                "SELECT state_json FROM experiment_data WHERE experiment_id = :id"
            ), {"id": experiment_id})).fetchone()
            if row and row[0]:
                await conn.execute(text("UPDATE session_state SET state_json = :s WHERE id = 1"), {"s": row[0]})

    # ── negative hypotheses ───────────────────────────────────────────────────

    def add_negative_hypothesis(self, hypothesis_text: str, status: str, research_question: str = "", analysis_summary: str = "", context_json: Optional[str] = None) -> None:
        _run(self._add_negative_hypothesis_async(hypothesis_text, status, research_question, analysis_summary, context_json))

    async def _add_negative_hypothesis_async(self, hypothesis_text: str, status: str, research_question: str, analysis_summary: str, context_json: Optional[str]) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO negative_hypotheses (hypothesis_text, status, research_question, analysis_summary, context_json, created_at) VALUES (:ht, :st, :rq, :as_, :cj, :ca)"
            ), {"ht": hypothesis_text, "st": status, "rq": research_question or "", "as_": analysis_summary or "", "cj": context_json, "ca": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    def get_negative_hypotheses(self, limit: Optional[int] = None) -> List[Dict]:
        return _run(self._get_negative_hypotheses_async(limit))

    async def _get_negative_hypotheses_async(self, limit: Optional[int]) -> List[Dict]:
        async with self._engine.connect() as conn:
            sql = "SELECT hypothesis_text, status, research_question, analysis_summary, created_at FROM negative_hypotheses ORDER BY id DESC"
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            rows = (await conn.execute(text(sql))).fetchall()
            return [{"hypothesis_text": r[0], "status": r[1], "research_question": r[2] or "", "analysis_summary": r[3] or "", "created_at": r[4] or ""} for r in rows]

    # ── hypothesis outcomes ───────────────────────────────────────────────────

    def add_hypothesis_outcome(self, hypothesis_text: str, status: str, material_hint: str = "", evidence_summary: str = "", source: str = "orchestrator") -> None:
        _run(self._add_hypothesis_outcome_async(hypothesis_text, status, material_hint, evidence_summary, source))

    async def _add_hypothesis_outcome_async(self, hypothesis_text: str, status: str, material_hint: str, evidence_summary: str, source: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO hypothesis_outcomes (hypothesis_text, material_hint, status, evidence_summary, source, created_at) VALUES (:ht, :mh, :st, :es, :src, :ca)"
            ), {"ht": hypothesis_text, "mh": material_hint or "", "st": status, "es": evidence_summary or "", "src": source or "orchestrator", "ca": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    def get_hypothesis_outcomes(self, limit: Optional[int] = 200) -> List[Dict]:
        return _run(self._get_hypothesis_outcomes_async(limit))

    async def _get_hypothesis_outcomes_async(self, limit: Optional[int]) -> List[Dict]:
        async with self._engine.connect() as conn:
            sql = "SELECT hypothesis_text, material_hint, status, evidence_summary, source, created_at FROM hypothesis_outcomes ORDER BY id DESC"
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            rows = (await conn.execute(text(sql))).fetchall()
            return [{"hypothesis_text": r[0], "material_hint": r[1] or "", "status": r[2], "evidence_summary": r[3] or "", "source": r[4] or "", "created_at": r[5] or ""} for r in rows]
