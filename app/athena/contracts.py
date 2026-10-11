"""ATHENA contracts used by the POLARIS simulator and execution orchestration layer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass
class AthenaCapability:
    capability_id: str
    name: str
    instrument_id: str
    owner_team: str
    profile_version: str = "athena-v1"
    mcp_endpoint: str | None = None
    supports_simulation: bool = True
    supports_live_execution: bool = False
    safety_class: str = "review_required"
    constraints: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AthenaExperimentBundle:
    experiment_id: str
    protocol_name: str
    protocol_source: str
    worklist_csv: str | None = None
    literature_context: dict[str, Any] = field(default_factory=dict)
    mode: str = "simulate"
    requested_by: str | None = None
    requested_at: str = field(default_factory=lambda: _utcnow().isoformat())

    def protocol_sha(self) -> str:
        return _sha256(self.protocol_source)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AthenaRunEvent:
    at: str = field(default_factory=lambda: _utcnow().isoformat())
    stage: str = ""
    actor: str = "polaris"
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AthenaValidationResult:
    mode: str
    status: str
    protocol_sha: str
    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: _utcnow().isoformat())
    finished_at: str = field(default_factory=lambda: _utcnow().isoformat())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AthenaRunRecord:
    run_id: str
    capability_id: str
    status: str
    bundle: AthenaExperimentBundle
    execution_mode: str
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: _utcnow().isoformat())
    validation: AthenaValidationResult | None = None
    events: list[AthenaRunEvent] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def mark(self, status: str, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self.updated_at = _utcnow().isoformat()
        self.events.append(
            AthenaRunEvent(
                stage=stage,
                message=message,
                payload=dict(payload or {}),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bundle"] = self.bundle.as_dict()
        data["validation"] = self.validation.as_dict() if self.validation else None
        return data
