"""Authenticated loopback HTTP service used by the browser extension."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
import secrets
from socketserver import TCPServer
import sys
from typing import Any
from urllib.parse import urlsplit

from .backend import (
    BackendConfig,
    BrowserInferenceBackend,
    BrowserInferenceError,
    DemoFixtureBrowserInferenceBackend,
    PrismGuardBrowserInferenceBackend,
)


SERVER_VERSION = "TechJamAIGC/0.2"
ALLOWED_REQUEST_HEADERS = (
    "Authorization, Content-Type, X-AIGC-Explain, X-AIGC-Artifact, "
    "X-AIGC-Source-Kind"
)


class BrowserThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def server_bind(self) -> None:
        """Bind without HTTPServer's unnecessary reverse-DNS lookup."""

        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


def is_allowed_origin(origin: str | None) -> bool:
    """Allow CLI clients without Origin and browser-extension/loopback origins."""

    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if parsed.scheme in {"chrome-extension", "moz-extension"}:
        return bool(parsed.hostname)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}


def _handler_class(backend: Any, api_token: str):
    class BrowserRequestHandler(BaseHTTPRequestHandler):
        server_version = SERVER_VERSION
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.client_address[0]} - {format % args}", file=sys.stderr)

        def _origin(self) -> str | None:
            return self.headers.get("Origin")

        def _base_headers(self, *, content_length: int) -> None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            origin = self._origin()
            if origin and is_allowed_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.close_connection = True
            self.send_response(status)
            self._base_headers(content_length=len(body))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str) -> None:
            self.close_connection = True
            self._json(status, {"error": {"code": code, "message": message}})

        def _request_allowed(self) -> bool:
            if not is_allowed_origin(self._origin()):
                self._error(403, "origin_forbidden", "Request origin is not allowed")
                return False
            return True

        def _authorized(self) -> bool:
            value = self.headers.get("Authorization", "")
            supplied = value[7:] if value.startswith("Bearer ") else ""
            if not supplied or not hmac.compare_digest(supplied, api_token):
                self._error(401, "unauthorized", "A valid local bearer token is required")
                return False
            return True

        def do_OPTIONS(self) -> None:  # noqa: N802
            if not self._request_allowed():
                return
            self.close_connection = True
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            origin = self._origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", ALLOWED_REQUEST_HEADERS)
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if not self._request_allowed() or not self._authorized():
                return
            if urlsplit(self.path).path != "/v1/health":
                self._error(404, "not_found", "Endpoint does not exist")
                return
            self._json(200, backend.health())

        def do_POST(self) -> None:  # noqa: N802
            if not self._request_allowed() or not self._authorized():
                return
            if urlsplit(self.path).path != "/v1/analyze":
                self._error(404, "not_found", "Endpoint does not exist")
                return
            raw_length = self.headers.get("Content-Length")
            try:
                content_length = int(raw_length or "")
            except ValueError:
                self._error(411, "content_length_required", "A valid Content-Length is required")
                return
            if content_length <= 0:
                self._error(400, "empty_image", "Image body is empty")
                return
            if content_length > backend.config.max_upload_bytes:
                self._error(
                    413,
                    "image_too_large",
                    f"Image exceeds the {backend.config.max_upload_bytes} byte upload limit",
                )
                return
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                self._error(400, "incomplete_body", "Image body ended before Content-Length")
                return
            explain = self.headers.get("X-AIGC-Explain", "").lower() in {
                "1",
                "true",
                "yes",
            }
            artifacts = self.headers.get("X-AIGC-Artifact", "").lower() in {
                "1",
                "true",
                "yes",
            }
            source_kind = self.headers.get("X-AIGC-Source-Kind", "viewport_capture")
            try:
                result = backend.analyze(
                    body,
                    media_type=media_type,
                    include_physics=explain,
                    include_artifacts=artifacts,
                    source_kind=source_kind,
                )
            except BrowserInferenceError as exc:
                self._error(exc.http_status, exc.code, str(exc))
                return
            except Exception as exc:  # pragma: no cover - production safety boundary
                print(f"inference error: {exc}", file=sys.stderr)
                self._error(500, "inference_failed", "Local inference failed")
                return
            self._json(200, result)

    return BrowserRequestHandler


def create_server(
    backend: Any,
    *,
    api_token: str,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> BrowserThreadingHTTPServer:
    if not api_token:
        raise ValueError("api_token cannot be empty")
    return BrowserThreadingHTTPServer(
        (host, port), _handler_class(backend, api_token)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m browser_product.service",
        description="Run the authenticated local AIGC inference service.",
    )
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint", help="Legacy pooled PatchHead checkpoint")
    model.add_argument(
        "--prismguard-bundle",
        help="Sealed PrismGuard pure-DINO detector bundle",
    )
    model.add_argument(
        "--demo-fixture",
        action="store_true",
        help="Run a deterministic wiring fixture with no AIGC performance claim",
    )
    parser.add_argument("--prismguard-checkpoint")
    parser.add_argument("--prismguard-license-ledger")
    parser.add_argument(
        "--prismguard-root",
        help="Optional PrismGuard repository root containing src/prismguard",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device")
    parser.add_argument("--api-token", default=os.environ.get("AIGC_EXTENSION_TOKEN"))
    parser.add_argument(
        "--physics-profile",
        choices=("off", "heuristic", "learned"),
        default="off",
    )
    parser.add_argument("--physics-cache-dir")
    parser.add_argument("--physics-offline", action="store_true")
    parser.add_argument("--strict-physics-models", action="store_true")
    parser.add_argument(
        "--artifact-profile",
        choices=("off", "residual"),
        default=None,
        help=(
            "Optional explanation-only residual sidecar (default: residual for "
            "PatchHead, off for PrismGuard)."
        ),
    )
    parser.add_argument("--artifact-checkpoint")
    parser.add_argument("--max-upload-mib", type=float, default=12.0)
    parser.add_argument("--max-pixels", type=int, default=25_000_000)
    parser.add_argument("--cache-entries", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must lie within [1, 65535]")
    if args.max_upload_mib <= 0 or args.max_upload_mib > 100:
        raise SystemExit("--max-upload-mib must lie within (0, 100]")
    token = args.api_token or secrets.token_urlsafe(32)
    artifact_profile = args.artifact_profile or (
        "off" if args.prismguard_bundle or args.demo_fixture else "residual"
    )
    config = BackendConfig(
        max_upload_bytes=int(args.max_upload_mib * 1024 * 1024),
        max_pixels=args.max_pixels,
        cache_entries=args.cache_entries,
        physics_profile=args.physics_profile,
        physics_cache_dir=args.physics_cache_dir,
        physics_offline=args.physics_offline,
        strict_physics_models=args.strict_physics_models,
        artifact_profile=artifact_profile,
        artifact_checkpoint=args.artifact_checkpoint,
        device=args.device,
    )
    try:
        if args.demo_fixture:
            if args.physics_profile != "off" or artifact_profile != "off":
                raise ValueError("demo fixture requires diagnostic profiles off")
            backend = DemoFixtureBrowserInferenceBackend(config=config)
        elif args.prismguard_bundle:
            if args.physics_profile != "off" or artifact_profile != "off":
                raise ValueError(
                    "PrismGuard browser mode requires --physics-profile off and "
                    "--artifact-profile off; diagnostics remain independent"
                )
            if args.prismguard_root:
                prismguard_source = os.path.join(
                    os.path.abspath(args.prismguard_root), "src"
                )
                if prismguard_source not in sys.path:
                    sys.path.insert(0, prismguard_source)
            backend = PrismGuardBrowserInferenceBackend(
                args.prismguard_bundle,
                checkpoint=args.prismguard_checkpoint,
                license_ledger=args.prismguard_license_ledger,
                config=config,
            )
        else:
            backend = BrowserInferenceBackend(args.checkpoint, config=config)
        server = create_server(
            backend,
            api_token=token,
            host=args.host,
            port=args.port,
        )
    except Exception as exc:
        print(f"browser-service: {exc}", file=sys.stderr)
        return 2
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: service is not bound to loopback; protect it with host firewall rules",
            file=sys.stderr,
        )
    print(f"AIGC browser service ready at http://{args.host}:{args.port}")
    print(f"Extension bearer token: {token}")
    print(f"Physics profile: {args.physics_profile}")
    print(f"Artifact profile: {artifact_profile}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping AIGC browser service")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
