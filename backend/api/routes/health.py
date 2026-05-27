"""
Health check API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.api.schemas import LiveResponse, ReadyResponse


router = APIRouter(tags=["health"])


def get_api_state(request: Request):
    return request.app.state.api_state


@router.get("/health/live", response_model=LiveResponse)
def live_check():
    return LiveResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse)
def readiness_check(api_state=Depends(get_api_state)):
    return ReadyResponse(
        ready=api_state.ready,
        detail=None if api_state.ready else api_state.ready_error,
    )
