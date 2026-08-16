"""End-to-end test: create a library, upload fixtures, run the pipeline as a
real subprocess, poll to completion, and fetch the Reads tab.
"""
import time
from pathlib import Path

import pytest

from app.services.phase_mapper_service import PhaseMapperService
from app.tools.database import DatabaseManager

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from app.db import engine as engine_module
    engine_module._engine = None
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))
    db = DatabaseManager()
    db.init_schema()
    yield PhaseMapperService(db)
    engine_module._engine = None


@pytest.mark.slow
def test_full_library_pipeline_round_trip(service):
    library = service.create_library("user-1", "Fixture Library")

    import io
    from fastapi import UploadFile

    async def _upload(path: Path, save_fn):
        upload = UploadFile(filename=path.name, file=io.BytesIO(path.read_bytes()))
        await save_fn(library["id"], "user-1", "composition" if "composition" in path.name else "plate_reader", upload)

    import asyncio
    asyncio.run(_upload(_FIXTURES / "composition_map_fixture.csv", service.save_upload))
    asyncio.run(_upload(_FIXTURES / "plate_reader_fixture.csv", service.save_upload))

    run = service.start_run(library["id"], "user-1")
    assert run["status"] == "running"

    deadline = time.time() + 60
    while time.time() < deadline:
        run = service.get_run(run["id"], "user-1")
        if run["status"] != "running":
            break
        time.sleep(1)

    assert run["status"] == "succeeded", run["error_message"]

    reads = service.get_reads(run["id"], "user-1")
    assert len(reads) >= 1
    assert reads[0]["read_type"] == "absorbance"
