"""Subprocess entry point for a phase-mapper pipeline run.

Mirrors app/tools/literature_agent_job_runner.py: reads a JSON payload
describing the run, executes it, and writes a returncode file the
polling service can observe without keeping a process handle open.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from app.services.phase_mapper.pipeline import run_pipeline


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"])
    return_code = 0
    try:
        run_pipeline(payload["config"], config_path=None)
    except Exception:
        return_code = 1
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    Path(payload["job_dir"], f"{payload['job_id']}.returncode").write_text(str(return_code), encoding="utf-8")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
