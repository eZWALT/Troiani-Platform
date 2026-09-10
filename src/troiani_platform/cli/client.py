from __future__ import annotations

from typing import Any

import httpx
import yaml

from troiani_platform.config import PlatformConfig


class PlatformClient:
    def __init__(self, config: PlatformConfig) -> None:
        headers = {}
        if config.control.token:
            headers["Authorization"] = f"Bearer {config.control.token}"
        self.http = httpx.Client(base_url=config.control.public_url.rstrip("/"), timeout=15.0, headers=headers)

    def get(self, path: str) -> Any:
        response = self.http.get(path)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = self.http.post(path, json=payload or {})
        response.raise_for_status()
        return response.json()

    def put(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = self.http.put(path, json=payload or {})
        response.raise_for_status()
        return response.json()

    def submit_file(self, path: str) -> Any:
        raw = yaml.safe_load(open(path, encoding="utf-8"))
        return self.post("/v1/jobs/from-yaml", raw if isinstance(raw, dict) else {"yaml": str(raw)})
