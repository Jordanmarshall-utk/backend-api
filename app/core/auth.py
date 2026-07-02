from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jwt
from fastapi import HTTPException, status

from app.core.config import Settings


@dataclass
class AuthUser:
    id: str
    email: str | None = None
    claims: Dict[str, Any] | None = None


def verify_supabase_jwt(token: str, settings: Settings) -> AuthUser:
    if not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth not configured (SUPABASE_JWT_SECRET)",
        )
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=settings.supabase_jwt_audience,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
    return AuthUser(id=str(sub), email=payload.get("email"), claims=payload)
