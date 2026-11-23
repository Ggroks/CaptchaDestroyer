"""HTTP entry point for the local four-digit image recognizer."""

import asyncio
from dataclasses import dataclass
from http import HTTPStatus
import logging
import math
import os
from pathlib import Path
import re
from secrets import compare_digest
import subprocess
import sys
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, StringConstraints
from python_multipart.exceptions import MultipartParseError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from ocr_worker import WorkerExit


LOGGER = logging.getLogger("uvicorn.error")
PROJECT_DIR = Path(__file__).resolve().parent
MULTIPART_OVERHEAD = 64 * 1024
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
IMAGE_MEDIA_TYPES = {
    "application/octet-stream", "image/bmp", "image/jpeg", "image/png", "image/webp", "image/x-ms-bmp",
}
WORKER_ERRORS = {
    WorkerExit.UNSUPPORTED_IMAGE: (415, "Unsupported image format"),
    WorkerExit.INVALID_IMAGE: (422, "Invalid image"),
    WorkerExit.IMAGE_TOO_LARGE: (413, "Image dimensions exceed the limit"),
    WorkerExit.RECOGNITION_FAILED: (422, "Recognition failed"),
}


@dataclass(frozen=True)
class Settings:
    api_key: str
    max_upload_bytes: int = 5 * 1024 * 1024
    timeout_seconds: float = 10
    max_concurrency: int = 2
    tessdata: Path = PROJECT_DIR / "tessdata"

    def __post_init__(self) -> None:
        if not self.api_key or any(not 33 <= ord(char) <= 126 for char in self.api_key):
            raise ValueError("OCR_API_KEY must contain printable ASCII characters without spaces")
        if self.max_upload_bytes < 1 or self.max_concurrency < 1:
            raise ValueError("OCR_MAX_UPLOAD_BYTES and OCR_MAX_CONCURRENCY must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("OCR_TIMEOUT_SECONDS must be positive and finite")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.environ.get("OCR_API_KEY", ""),
            max_upload_bytes=int(os.environ.get("OCR_MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
            timeout_seconds=float(os.environ.get("OCR_TIMEOUT_SECONDS", cls.timeout_seconds)),
            max_concurrency=int(os.environ.get("OCR_MAX_CONCURRENCY", cls.max_concurrency)),
            tessdata=Path(os.environ.get("OCR_TESSDATA_DIR") or cls.tessdata).resolve(),
        )


class Recognized(BaseModel):
    success: Literal[True] = True
    text: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
    processing_ms: int


class RecognitionError(BaseModel):
    success: Literal[False] = False
    text: None = None
    error: str


def failure(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=RecognitionError(error=message).model_dump(),
        headers=headers,
    )


def run_worker(data: bytes, settings: Settings) -> subprocess.CompletedProcess[bytes]:
    # subprocess.run kills and waits for the child before it reports a timeout.
    return subprocess.run(
        [sys.executable, "-m", "ocr_worker", str(settings.tessdata)],
        cwd=PROJECT_DIR,
        input=data,
        capture_output=True,
        timeout=settings.timeout_seconds,
        check=False,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings.from_env()
    expected_key = settings.api_key.encode("ascii")
    slots = asyncio.Semaphore(settings.max_concurrency)
    model = settings.tessdata / "eng.traineddata"
    app = FastAPI(
        title="Digit OCR API",
        max_body_size=settings.max_upload_bytes + MULTIPART_OVERHEAD,
    )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, error: HTTPException) -> JSONResponse:
        message = {
            400: "Invalid multipart upload",
            413: "Image file exceeds the upload limit",
        }.get(error.status_code, HTTPStatus(error.status_code).phrase)
        return failure(error.status_code, message, error.headers)

    @app.get("/health", responses={503: {"model": RecognitionError}})
    async def health():
        if not model.is_file():
            return failure(503, "OCR model unavailable")
        return {"status": "ok"}

    @app.post(
        "/api/recognize",
        response_model=Recognized,
        responses={
            code: {"model": RecognitionError}
            for code in (400, 401, 413, 415, 422, 500, 503, 504)
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {"file": {"type": "string", "format": "binary"}},
                        }
                    }
                },
            }
        },
    )
    async def recognize(request: Request, key: Annotated[str | None, Security(API_KEY_HEADER)]):
        if not key or not compare_digest(key.encode("utf-8"), expected_key):
            return failure(401, "Invalid or missing API key", {"WWW-Authenticate": "APIKey"})
        content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type != "multipart/form-data":
            return failure(415, "Expected multipart/form-data")

        try:
            async with request.form(max_files=1, max_fields=0) as form:
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    return failure(400, "Expected an image in the file field")
                if upload.size is not None and upload.size > settings.max_upload_bytes:
                    return failure(413, "Image file exceeds the upload limit")
                part_type = (upload.content_type or "application/octet-stream").partition(";")[0]
                if part_type.strip().lower() not in IMAGE_MEDIA_TYPES:
                    return failure(415, "Unsupported image format")
                data = await upload.read(settings.max_upload_bytes + 1)
        except MultipartParseError:
            return failure(400, "Invalid multipart upload")

        if len(data) > settings.max_upload_bytes:
            return failure(413, "Image file exceeds the upload limit")
        if not data:
            return failure(422, "Invalid image")
        if slots.locked():
            return failure(503, "OCR service busy", {"Retry-After": "1"})

        started = perf_counter()
        async with slots:
            job = asyncio.create_task(asyncio.to_thread(run_worker, data, settings))
            try:
                result = await asyncio.shield(job)
            except asyncio.CancelledError:
                # Keep this slot occupied until the child exits or is killed on timeout.
                await asyncio.gather(job, return_exceptions=True)
                raise
            except subprocess.TimeoutExpired:
                return failure(504, "Recognition timed out")
            except OSError:
                LOGGER.exception("Could not run OCR process")
                return failure(500, "Recognition failed")

        if result.returncode in WORKER_ERRORS:
            return failure(*WORKER_ERRORS[result.returncode])
        text = result.stdout.strip()
        if result.returncode or not re.fullmatch(rb"[0-9]{4}", text):
            LOGGER.error("OCR process failed (%s): %r", result.returncode, result.stderr[-4096:])
            return failure(500, "Recognition failed")
        processing_ms = round((perf_counter() - started) * 1000)
        return Recognized(text=text.decode("ascii"), processing_ms=processing_ms)

    return app
