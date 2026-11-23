import asyncio
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import unittest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw, ImageFont

from ocr_api import MULTIPART_OVERHEAD, Settings, create_app, run_worker


SETTINGS = Settings(api_key="test-key")
AUTH = {"X-API-Key": SETTINGS.api_key}


def image_bytes(text: str = "", format: str = "PNG") -> bytes:
    image = Image.new("RGB", (320, 140), "white")
    font = ImageFont.load_default(size=88)
    ImageDraw.Draw(image).text((160, 70), text, font=font, fill="black", anchor="mm")
    with BytesIO() as buffer:
        image.save(buffer, format=format)
        return buffer.getvalue()


class TestOcrApi(unittest.IsolatedAsyncioTestCase):
    def client(self, settings: Settings = SETTINGS) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=create_app(settings)),
            base_url="http://test",
        )

    async def recognize(self, client, data, headers=AUTH, media_type="image/png"):
        return await client.post(
            "/api/recognize",
            headers=headers,
            files={"file": ("image", data, media_type)},
        )

    def assert_error(self, response, status, error):
        self.assertEqual(response.status_code, status)
        self.assertEqual(response.json(), {"success": False, "text": None, "error": error})

    async def test_success_and_leading_zeroes(self):
        async with self.client() as client:
            response = await self.recognize(client, image_bytes("0042"))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        processing_ms = result.pop("processing_ms")
        self.assertEqual(result, {"success": True, "text": "0042"})
        self.assertIs(type(processing_ms), int)
        self.assertGreaterEqual(processing_ms, 0)

    async def test_missing_api_key(self):
        async with self.client() as client:
            response = await self.recognize(client, image_bytes("0042"), headers={})
        self.assert_error(response, 401, "Invalid or missing API key")
        self.assertEqual(response.headers["WWW-Authenticate"], "APIKey")

    async def test_incorrect_api_key(self):
        async with self.client() as client:
            response = await self.recognize(client, image_bytes("0042"), headers={"X-API-Key": "wrong"})
        self.assert_error(response, 401, "Invalid or missing API key")

    async def test_missing_file_field(self):
        async with self.client() as client:
            response = await client.post(
                "/api/recognize",
                headers=AUTH,
                files={"wrong_field": ("image.png", image_bytes("0042"), "image/png")},
            )
        self.assert_error(response, 400, "Expected an image in the file field")

    async def test_unsupported_format_checks_actual_image(self):
        async with self.client() as client:
            response = await self.recognize(client, image_bytes("0042", format="GIF"))
        self.assert_error(response, 415, "Unsupported image format")

    async def test_corrupt_image(self):
        async with self.client() as client:
            response = await self.recognize(client, image_bytes("0042")[:24])
        self.assert_error(response, 422, "Invalid image")

    async def test_no_recognition_result(self):
        async with self.client() as client:
            response = await self.recognize(client, image_bytes())
        self.assert_error(response, 422, "Recognition failed")

    async def test_recognizer_error_is_hidden(self):
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "eng.traineddata").write_bytes(b"invalid model used only in tests")
            async with self.client(replace(SETTINGS, tessdata=model_dir)) as client:
                with self.assertLogs("uvicorn.error", level="ERROR"):
                    response = await self.recognize(client, image_bytes("0042"))
        self.assert_error(response, 500, "Recognition failed")

    async def test_concurrent_requests(self):
        both_workers_started = Barrier(2, timeout=5)

        def run_concurrently(data, settings):
            both_workers_started.wait()
            return run_worker(data, settings)

        transport = ASGITransport(app=create_app(SETTINGS))
        async with (
            AsyncClient(transport=transport, base_url="http://test") as first_client,
            AsyncClient(transport=transport, base_url="http://test") as second_client,
        ):
            with patch("ocr_api.run_worker", side_effect=run_concurrently):
                first, second = await asyncio.gather(
                    self.recognize(first_client, image_bytes("0042")),
                    self.recognize(second_client, image_bytes("5831")),
                )
        self.assertEqual([first.status_code, second.status_code], [200, 200])
        self.assertEqual([first.json()["text"], second.json()["text"]], ["0042", "5831"])

    async def test_upload_size_limit(self):
        async with self.client(replace(SETTINGS, max_upload_bytes=64)) as client:
            response = await self.recognize(client, image_bytes("0042"))
        self.assert_error(response, 413, "Image file exceeds the upload limit")

    async def test_service_busy_and_health_stays_available(self):
        entered = Event()
        release = Event()

        def run_after_release(data, settings):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the OCR worker")
            return run_worker(data, settings)

        async with self.client(replace(SETTINGS, max_concurrency=1)) as client:
            with patch("ocr_api.run_worker", side_effect=run_after_release):
                active = asyncio.create_task(self.recognize(client, image_bytes("0042")))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                    response = await self.recognize(client, image_bytes("5831"))
                    self.assert_error(response, 503, "OCR service busy")
                    self.assertEqual(response.headers["Retry-After"], "1")
                    self.assertEqual((await client.get("/health")).json(), {"status": "ok"})
                finally:
                    release.set()
                    self.assertEqual((await active).status_code, 200)

    async def test_streamed_request_size_limit(self):
        async def body():
            yield b"--fixture\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.png\"\r\n\r\n"
            for _ in range(3):
                yield b"0" * MULTIPART_OVERHEAD
            yield b"\r\n--fixture--\r\n"

        async with self.client(replace(SETTINGS, max_upload_bytes=64)) as client:
            response = await client.post(
                "/api/recognize",
                content=body(),
                headers={**AUTH, "Content-Type": "multipart/form-data; boundary=fixture"},
            )
        self.assert_error(response, 413, "Image file exceeds the upload limit")

    async def test_timeout_and_slot_reuse(self):
        async with self.client(replace(SETTINGS, timeout_seconds=0.01, max_concurrency=1)) as client:
            for _ in range(2):
                response = await self.recognize(client, image_bytes("0042"))
                self.assert_error(response, 504, "Recognition timed out")
            self.assertEqual((await client.get("/health")).json(), {"status": "ok"})

    async def test_health_with_missing_model(self):
        with TemporaryDirectory() as directory:
            async with self.client(replace(SETTINGS, tessdata=Path(directory))) as client:
                response = await client.get("/health")
        self.assert_error(response, 503, "OCR model unavailable")


if __name__ == "__main__":
    unittest.main()
