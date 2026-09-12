from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any


class AuthError(ValueError):
    pass


class AuthStore:
    def __init__(self, csv_path: str | Path, secret: str, ttl_seconds: int = 2592000) -> None:
        self.csv_path = Path(csv_path)
        self.secret = secret.encode("utf-8")
        self.ttl_seconds = ttl_seconds

    def users(self) -> dict[str, dict[str, str]]:
        try:
            with self.csv_path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != [
                    "code",
                    "name",
                    "role",
                    "default_inference_mode",
                ]:
                    raise AuthError(
                        "auth.csv must contain exactly "
                        "code,name,role,default_inference_mode columns"
                    )
                rows = list(reader)
        except FileNotFoundError as exc:
            raise AuthError("auth.csv not found") from exc
        except UnicodeDecodeError as exc:
            raise AuthError("auth.csv must be UTF-8 CSV") from exc
        except OSError as exc:
            raise AuthError(f"auth.csv unavailable: {exc}") from exc

        users: dict[str, dict[str, str]] = {}
        for row in rows:
            code = str(row.get("code") or "").strip()
            name = str(row.get("name") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            default_inference_mode = (
                str(row.get("default_inference_mode") or "").strip().lower()
                or "offline"
            )
            if not code or not name:
                continue
            if role not in {"admin", "user"}:
                raise AuthError(f"invalid role in auth.csv: {role}")
            if default_inference_mode not in {"online", "offline"}:
                raise AuthError(
                    "invalid default_inference_mode in auth.csv: "
                    f"{default_inference_mode}"
                )
            if code in users:
                raise AuthError(f"duplicate code in auth.csv: {code}")
            users[code] = {
                "code": code,
                "name": name,
                "role": role,
                "default_inference_mode": default_inference_mode,
            }
        if not users:
            raise AuthError("auth.csv has no users")
        return users

    def login(self, code: str) -> tuple[dict[str, str], str]:
        user = self.users().get(str(code).strip())
        if not user:
            raise AuthError("invalid login code")
        payload = {
            "code": user["code"],
            "exp": int(time.time()) + self.ttl_seconds,
        }
        body = self._encode(payload)
        signature = self._sign(body)
        return user, f"{body}.{signature}"

    def verify(self, token: str) -> dict[str, str]:
        try:
            body, signature = token.split(".", 1)
            expected = self._sign(body)
            if not hmac.compare_digest(signature, expected):
                raise AuthError("invalid session")
            payload = json.loads(self._decode(body))
            if int(payload["exp"]) < int(time.time()):
                raise AuthError("session expired")
            user = self.users().get(str(payload["code"]))
            if not user:
                raise AuthError("user no longer exists")
            return user
        except (AuthError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, AuthError):
                raise
            raise AuthError("invalid session") from exc

    def _sign(self, body: str) -> str:
        return base64.urlsafe_b64encode(
            hmac.new(self.secret, body.encode("ascii"), hashlib.sha256).digest()
        ).decode("ascii").rstrip("=")

    @staticmethod
    def _encode(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode(value: str) -> str:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("ascii")).decode("utf-8")
