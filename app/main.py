"""HTTP layer for the phasing service: POST /api/phase and /health."""

from __future__ import annotations

import time

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .solver import PhaseError, phase

app = FastAPI(
    title="Ancient-DNA phasing API",
    description=(
        "Jointly recovers a pair of complementary binary haplotypes and a "
        "unique group assignment per molecular read."
    ),
    version="1.0.0",
)

_STARTED_AT = time.time()


@app.exception_handler(PhaseError)
async def phase_error_handler(_: Request, exc: PhaseError) -> JSONResponse:
    status = 422 if exc.code == "INVALID_INPUT" else 409
    return JSONResponse(
        status_code=status,
        content={
            "ok": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
            },
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "uptime_s": round(time.time() - _STARTED_AT, 3)}


@app.post("/api/phase")
async def api_phase(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "BAD_JSON", "message": "request body must be valid JSON"}},
        )
    # Solving is CPU-bound (up to ~3s on worst-case input); run it off the
    # event loop so /health stays responsive.
    result = await anyio.to_thread.run_sync(phase, body)
    return JSONResponse(status_code=200, content={"ok": True, "data": result})
