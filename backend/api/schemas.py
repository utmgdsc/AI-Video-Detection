"""
Pydantic schemas for backend API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    result: dict[str, Any]


class LiveResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    detail: str | None = None
