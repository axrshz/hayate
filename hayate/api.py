"""Provide an optional FastAPI interface for streamed text generation."""

import argparse
import asyncio
import json
import threading
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from hayate.engine.engine import Engine
from hayate.engine.request import Request as GenerationRequest


class GenerateBody(BaseModel):
    """Describe a prompt and its sampling controls."""

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=100, ge=1, strict=True)
    temperature: float = Field(default=0.0, ge=0.0, strict=True)
    top_k: int | None = Field(default=None, ge=1, strict=True)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, strict=True)


class EngineRuntime:
    """Serialize access to one engine and track its HTTP requests."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.registry_lock = threading.Lock()
        self.requests: dict[int, GenerationRequest] = {}

    def submit(self, body: GenerateBody) -> GenerationRequest:
        """Prepare one prompt and add it to the engine scheduler."""
        with self.lock:
            request = self.engine.submit_request(
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
            )
            with self.registry_lock:
                self.requests[request.id] = request
            return request

    def advance(self) -> bool:
        """Run one engine step."""
        with self.lock:
            return self.engine.generate()

    def observe(self, request: GenerationRequest, after_token_count: int) -> dict:
        """Copy request state and build any new token events."""
        with self.lock:
            tokens = list(request.tokens)
            token_events = []
            for index in range(after_token_count, len(tokens)):
                token_events.append(
                    {
                        "token_id": tokens[index],
                        "token_count": index + 1,
                        "text": self.engine.tokenizer.decode(
                            tokens[: index + 1], skip_special_tokens=True
                        ),
                    }
                )
            text = request.response
            if text is None:
                text = self.engine.tokenizer.decode(
                    tokens, skip_special_tokens=True
                )
            return {
                "token_events": token_events,
                "token_count": len(tokens),
                "text": text,
                "is_completed": request.is_completed,
                "is_cancelled": request.is_cancelled,
            }

    def cancel(self, request_id: int) -> dict | None:
        """Cancel one tracked request, if it is still active."""
        with self.registry_lock:
            request = self.requests.get(request_id)
        if request is None:
            return None
        if not request.is_completed:
            # This flag is visible while the engine lock protects an active GPU step.
            request.is_cancel_requested = True

        with self.lock:
            with self.registry_lock:
                if self.requests.get(request_id) is not request:
                    return None
            if request.is_completed:
                cancelled = request.is_cancelled
            else:
                cancelled = self.engine.cancel_request(request)
            status = "cancelled" if cancelled else "completed"
            return {
                "request_id": request.id,
                "status": status,
                "text": request.response or "",
            }

    def close(self, request: GenerationRequest) -> None:
        """Cancel unfinished work and remove the request from the API registry."""
        with self.lock:
            if not request.is_completed:
                self.engine.cancel_request(request)
            with self.registry_lock:
                if self.requests.get(request.id) is request:
                    del self.requests[request.id]


def _event(name: str, payload: dict) -> str:
    """Format one Server-Sent Event."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n"


async def _stream_request(
    engine_runtime: EngineRuntime,
    generation_request: GenerationRequest,
) -> AsyncIterator[str]:
    """Advance the engine and send request events until completion."""
    request_id = generation_request.id
    sent_token_count = 0
    advance_task: asyncio.Task[bool] | None = None
    yield _event("started", {"request_id": request_id})

    try:
        while True:
            state = await asyncio.to_thread(
                engine_runtime.observe, generation_request, sent_token_count
            )
            for token_event in state["token_events"]:
                sent_token_count = token_event["token_count"]
                yield _event("token", {"request_id": request_id, **token_event})

            if state["is_completed"]:
                event_name = "cancelled" if state["is_cancelled"] else "completed"
                yield _event(
                    event_name,
                    {
                        "request_id": request_id,
                        "token_count": state["token_count"],
                        "text": state["text"],
                    },
                )
                return

            advance_task = asyncio.create_task(
                asyncio.to_thread(engine_runtime.advance)
            )
            while not advance_task.done():
                done, _ = await asyncio.wait({advance_task}, timeout=15.0)
                if not done:
                    yield ": keep-alive\n\n"
            had_work = await advance_task
            advance_task = None
            if not had_work:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.01)
    except Exception as exc:
        yield _event(
            "error", {"request_id": request_id, "message": str(exc)}
        )
    finally:
        if advance_task is not None and not advance_task.done():
            # Cancel the asyncio wrapper. A worker thread cannot stop mid-step.
            advance_task.cancel()
        await asyncio.shield(
            asyncio.to_thread(engine_runtime.close, generation_request)
        )


def create_app(engine: Engine) -> FastAPI:
    """Create an HTTP app for a ready inference engine."""
    app = FastAPI(title="Hayate inference engine")
    engine_runtime = EngineRuntime(engine)
    app.state.engine_runtime = engine_runtime

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(body: GenerateBody) -> StreamingResponse:
        try:
            generation_request = await asyncio.to_thread(
                engine_runtime.submit, body
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return StreamingResponse(
            _stream_request(engine_runtime, generation_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(engine_runtime.close, generation_request),
        )

    @app.post("/requests/{request_id}/cancel")
    async def cancel(request_id: int) -> dict:
        result = await asyncio.to_thread(engine_runtime.cancel, request_id)
        if result is None:
            raise HTTPException(status_code=404, detail="request not found")
        return result

    return app


def main() -> None:
    """Load a model and start the HTTP server."""
    parser = argparse.ArgumentParser(description="Run the Hayate HTTP server.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    engine = Engine(args.model)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
