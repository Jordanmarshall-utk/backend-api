from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_DEFAULT_URL = "sqlite+aiosqlite:///./data/polaris.db"
_engine: AsyncEngine | None = None


def get_db_url() -> str:
    raw = os.environ.get("DATABASE_URL", "").strip()
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    if raw.startswith("postgresql+psycopg://"):
        raw = raw.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if raw.startswith("postgresql+psycopg2://"):
        raw = raw.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if raw.startswith("postgresql+asyncpg://") or raw.startswith("sqlite+aiosqlite://"):
        return raw
    if raw.startswith("sqlite://"):
        return raw.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if raw:
        raise ValueError(
            "Unsupported DATABASE_URL scheme. Use PostgreSQL or SQLite with an async driver."
        )
    db_path = os.environ.get("POLARIS_DB_PATH", "").strip()
    if db_path:
        resolved = Path(db_path).expanduser().resolve()
        return f"sqlite+aiosqlite:///{resolved.as_posix()}"
    return _DEFAULT_URL


def get_async_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = get_db_url()
        if url.startswith("sqlite"):
            _engine = create_async_engine(url, connect_args={"check_same_thread": False})
        else:
            _engine = create_async_engine(url, pool_size=5, max_overflow=10, pool_pre_ping=True)
    return _engine
