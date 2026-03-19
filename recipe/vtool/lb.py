#!/usr/bin/env python3
import argparse
import asyncio
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator
import uvicorn


class RatioUpdate(BaseModel):
    ratio: str

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: str) -> str:
        parse_ratio(v)
        return v


def parse_ratio(ratio: str) -> Tuple[int, int]:
    try:
        a, b = ratio.split(":")
        w1, w2 = int(a), int(b)
        if w1 < 0 or w2 < 0:
            raise ValueError
        if w1 == 0 and w2 == 0:
            raise ValueError
        return w1, w2
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio '{ratio}'. Expected format like 3:1, 0:4, or 4:0, with at least one side > 0"
        ) from e


def strip_hop_by_hop_headers(headers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
    return {k: v for k, v in headers if k.lower() not in hop_by_hop}


class DynamicWeightedRoundRobin:
    def __init__(self, endpoints: List[str], weights: List[int]):
        if len(endpoints) != 2 or len(weights) != 2:
            raise ValueError("Exactly two endpoints and two weights are required.")

        self._endpoints = [e.rstrip("/") for e in endpoints]
        self._weights = [0, 0]
        self._queue: Deque[str] = deque()
        self._lock = asyncio.Lock()
        self._set_weights_unlocked(weights)

    def _set_weights_unlocked(self, weights: List[int]) -> None:
        if len(weights) != 2:
            raise ValueError("Exactly two weights are required.")
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative integers.")
        if sum(weights) == 0:
            raise ValueError("At least one weight must be greater than 0.")

        self._weights = list(weights)
        self._queue = deque()

        for endpoint, weight in zip(self._endpoints, self._weights):
            if weight > 0:
                self._queue.extend([endpoint] * weight)

    async def next(self) -> str:
        async with self._lock:
            if not self._queue:
                raise RuntimeError("No active upstream endpoints configured.")
            endpoint = self._queue[0]
            self._queue.rotate(-1)
            return endpoint

    async def set_ratio(self, ratio: str) -> None:
        w1, w2 = parse_ratio(ratio)
        async with self._lock:
            self._set_weights_unlocked([w1, w2])

    async def get_state(self) -> Dict[str, object]:
        async with self._lock:
            return {
                "endpoint1": self._endpoints[0],
                "endpoint2": self._endpoints[1],
                "weights": {
                    self._endpoints[0]: self._weights[0],
                    self._endpoints[1]: self._weights[1],
                },
                "ratio": f"{self._weights[0]}:{self._weights[1]}",
                "active_endpoints": [
                    endpoint
                    for endpoint, weight in zip(self._endpoints, self._weights)
                    if weight > 0
                ],
            }


def build_app(endpoint1: str, endpoint2: str, ratio: str) -> FastAPI:
    w1, w2 = parse_ratio(ratio)
    balancer = DynamicWeightedRoundRobin(
        endpoints=[endpoint1, endpoint2],
        weights=[w1, w2],
    )

    app = FastAPI()
    client = httpx.AsyncClient(timeout=None)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await client.aclose()

    @app.get("/admin/ratio")
    async def get_ratio():
        return await balancer.get_state()

    @app.post("/admin/ratio")
    async def update_ratio(payload: RatioUpdate):
        try:
            await balancer.set_ratio(payload.ratio)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "status": "ok",
            "message": "Ratio updated",
            "config": await balancer.get_state(),
        }

    async def proxy_request(request: Request, path: str):
        target_base = await balancer.next()
        target_url = f"{target_base}/{path}"

        query_params = request.url.query
        if query_params:
            target_url = f"{target_url}?{query_params}"

        body = await request.body()
        headers = strip_hop_by_hop_headers(request.headers.items())

        try:
            upstream_req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.RequestError as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Upstream request failed: {str(e)}"},
            )

        response_headers = strip_hop_by_hop_headers(upstream_resp.headers.items())

        content_type = upstream_resp.headers.get("content-type", "")
        is_streaming = (
            "text/event-stream" in content_type
            or request.headers.get("accept") == "text/event-stream"
        )

        if is_streaming:
            async def iter_stream():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()

            return StreamingResponse(
                iter_stream(),
                status_code=upstream_resp.status_code,
                headers=response_headers,
                media_type=upstream_resp.headers.get("content-type"),
            )

        try:
            content = await upstream_resp.aread()
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                headers=response_headers,
                media_type=upstream_resp.headers.get("content-type"),
            )
        finally:
            await upstream_resp.aclose()

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def catch_all(request: Request, path: str):
        if path.startswith("admin/"):
            raise HTTPException(status_code=404, detail="Not found")
        return await proxy_request(request, path)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Weighted load balancer for two vLLM endpoints with runtime ratio updates."
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument(
        "--endpoint1",
        required=True,
        help="First vLLM endpoint, e.g. http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--endpoint2",
        required=True,
        help="Second vLLM endpoint, e.g. http://127.0.0.1:8002",
    )
    parser.add_argument(
        "--ratio",
        required=True,
        help="Traffic ratio like 3:1, 0:4, or 4:0",
    )

    args = parser.parse_args()
    app = build_app(args.endpoint1, args.endpoint2, args.ratio)
    uvicorn.run(app, host=args.listen_host, port=args.listen_port)


if __name__ == "__main__":
    main()


# Optional import-time app for `uvicorn lb:app`
def create_app():
    return build_app(
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
        "2:2",
    )


app = create_app()