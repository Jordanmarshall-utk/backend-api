"""Unit tests for PhaseMapperService, exercised against a real DatabaseManager."""
import os

import pytest

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from app.services.phase_mapper_service import PhaseMapperService  # noqa: E402
from app.tools.database import DatabaseManager  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    # Fresh in-memory engine per test so schema/state don't leak across tests.
    from app.db import engine as engine_module
    engine_module._engine = None
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    manager = DatabaseManager()
    manager.init_schema()
    yield manager
    engine_module._engine = None


@pytest.fixture
def service(db):
    return PhaseMapperService(db)


def test_get_library_owned_returns_library_for_owner(service, db):
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")

    library = service.get_library_owned(library_id, "user-A")

    assert library["id"] == library_id
    assert library["user_id"] == "user-A"


def test_get_library_owned_raises_for_different_user(service, db):
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")

    with pytest.raises(LookupError):
        service.get_library_owned(library_id, "user-B")


def test_get_library_owned_raises_for_nonexistent_library(service):
    with pytest.raises(LookupError):
        service.get_library_owned(999999, "user-A")
