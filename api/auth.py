"""Clerk JWT verification.

Production: every request must carry an `Authorization: Bearer <jwt>` header
issued by Clerk. We verify the signature against Clerk's JWKS (cached) and
extract `sub` as the user_id.

Dev convenience: when CLERK_JWKS_URL is unset (or AUTH_DISABLED=1) we accept
any request and stamp it with user_id "anonymous". This keeps `localhost`
ergonomic while still letting prod be strict.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx
from fastapi import Depends, Header, HTTPException
from jose import jwt

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")  # e.g. https://YOUR-DOMAIN.clerk.accounts.dev/.well-known/jwks.json
CLERK_ISSUER = os.getenv("CLERK_ISSUER")  # e.g. https://YOUR-DOMAIN.clerk.accounts.dev
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "0") == "1"

_JWKS_CACHE: dict[str, object] = {"keys": None, "fetched_at": 0.0}
_JWKS_TTL = 60 * 60  # 1 hour


def _fetch_jwks() -> dict:
    now = time.time()
    cached = _JWKS_CACHE.get("keys")
    if cached and (now - float(_JWKS_CACHE["fetched_at"])) < _JWKS_TTL:
        return cached  # type: ignore[return-value]
    if not CLERK_JWKS_URL:
        raise HTTPException(500, "CLERK_JWKS_URL not configured")
    resp = httpx.get(CLERK_JWKS_URL, timeout=10)
    resp.raise_for_status()
    keys = resp.json()
    _JWKS_CACHE["keys"] = keys
    _JWKS_CACHE["fetched_at"] = now
    return keys


def _verify_clerk_jwt(token: str) -> str:
    try:
        unverified = jwt.get_unverified_header(token)
    except Exception as e:
        raise HTTPException(401, f"Bad token header: {e}")
    kid = unverified.get("kid")
    jwks = _fetch_jwks()
    keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
    matching = next((k for k in keys if k.get("kid") == kid), None)
    if not matching:
        raise HTTPException(401, "Signing key not found in JWKS")
    try:
        claims = jwt.decode(
            token,
            matching,
            algorithms=[matching.get("alg", "RS256")],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False},  # Clerk session tokens may not have aud
        )
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(401, "Token has no sub")
    return sub


async def get_current_user_id(
    authorization: Optional[str] = Header(default=None),
) -> str:
    """FastAPI dependency. Returns the verified Clerk user id, or "anonymous"
    in dev when auth is intentionally disabled.
    """
    if AUTH_DISABLED or not CLERK_JWKS_URL:
        return "anonymous"
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    return _verify_clerk_jwt(token)


# Re-export as a typed Depends shorthand callers can stash on `user_id: str = ...`.
CurrentUser = Depends(get_current_user_id)
