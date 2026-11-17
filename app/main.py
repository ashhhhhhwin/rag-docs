import logging
import uuid

from fastapi import FastAPI, Request

from app.api.routes import router as ask_router
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Chat With Your Docs")

app.include_router(ask_router)


@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    # Trace id on every request; we'll thread it through logs later
    request.state.trace_id = request.headers.get("x-trace-id", uuid.uuid4().hex)
    response = await call_next(request)
    response.headers["x-trace-id"] = request.state.trace_id
    return response


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm_model": s.llm_model, "embedding_model": s.embedding_model}
