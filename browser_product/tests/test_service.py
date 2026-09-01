from __future__ import annotations

from io import BytesIO
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from browser_product.backend import BackendConfig, BrowserInferenceBackend
from browser_product.service import create_server, is_allowed_origin
from patchhead.tests.test_inference import FakeFeatureRuntime


TOKEN = "fixture-local-token"
ORIGIN = "chrome-extension://abcdefghijklmnop"


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 60), (70, 100, 130)).save(output, format="PNG")
    return output.getvalue()


class BrowserServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        backend = BrowserInferenceBackend(
            runtime=FakeFeatureRuntime(),
            config=BackendConfig(cache_entries=4, max_upload_bytes=1024),
        )
        self.server = create_server(backend, api_token=TOKEN, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        token: str | None = TOKEN,
        origin: str = ORIGIN,
        content_type: str = "image/png",
        include_artifacts: bool = False,
    ):
        headers = {"Origin": origin}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if body is not None:
            headers["Content-Type"] = content_type
            headers["X-AIGC-Source-Kind"] = "test_fixture"
            headers["X-AIGC-Artifact"] = "true" if include_artifacts else "false"
        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=5) as response:
            return response, json.loads(response.read())

    def test_health_requires_token_and_reflects_extension_origin(self) -> None:
        with self.assertRaises(HTTPError) as captured:
            self._request("/v1/health", token=None)
        self.assertEqual(captured.exception.code, 401)

        response, payload = self._request("/v1/health")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], ORIGIN)
        self.assertEqual(payload["status"], "ready")

    def test_analyze_returns_result_and_bounded_cache_hit(self) -> None:
        first_response, first = self._request(
            "/v1/analyze", method="POST", body=_png_bytes()
        )
        _, second = self._request(
            "/v1/analyze", method="POST", body=_png_bytes()
        )

        self.assertEqual(first_response.status, 200)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["verdict"], second["verdict"])

    def test_artifact_request_header_is_reported_when_profile_is_off(self) -> None:
        _, payload = self._request(
            "/v1/analyze",
            method="POST",
            body=_png_bytes(),
            include_artifacts=True,
        )

        self.assertTrue(payload["explanation"]["artifact_requested"])
        self.assertIsNone(payload["explanation"]["artifact"])
        self.assertFalse(payload["explanation"]["artifact_affects_detector_score"])

    def test_untrusted_web_origin_is_rejected(self) -> None:
        with self.assertRaises(HTTPError) as captured:
            self._request("/v1/health", origin="https://malicious.example")
        self.assertEqual(captured.exception.code, 403)

    def test_upload_limit_and_media_type_fail_before_inference(self) -> None:
        with self.assertRaises(HTTPError) as oversized:
            self._request("/v1/analyze", method="POST", body=b"x" * 1025)
        self.assertEqual(oversized.exception.code, 413)

        with self.assertRaises(HTTPError) as media_type:
            self._request(
                "/v1/analyze",
                method="POST",
                body=_png_bytes(),
                content_type="application/octet-stream",
            )
        self.assertEqual(media_type.exception.code, 415)

    def test_cors_preflight_is_narrow_and_does_not_require_token(self) -> None:
        request = Request(
            self.base_url + "/v1/analyze",
            headers={"Origin": ORIGIN},
            method="OPTIONS",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], ORIGIN)
            self.assertEqual(
                response.headers["Access-Control-Allow-Methods"],
                "GET, POST, OPTIONS",
            )
            self.assertIn(
                "X-AIGC-Artifact", response.headers["Access-Control-Allow-Headers"]
            )

    def test_origin_parser_allows_only_extensions_and_loopback(self) -> None:
        self.assertTrue(is_allowed_origin(None))
        self.assertTrue(is_allowed_origin(ORIGIN))
        self.assertTrue(is_allowed_origin("moz-extension://abc123"))
        self.assertTrue(is_allowed_origin("http://127.0.0.1:9000"))
        self.assertFalse(is_allowed_origin("https://localhost"))
        self.assertFalse(is_allowed_origin("https://example.com"))


if __name__ == "__main__":
    unittest.main()
