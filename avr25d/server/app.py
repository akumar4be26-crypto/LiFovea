"""Zero-dependency HTTP server for the dashboard.

Serves the same ``web/dashboard.html`` that gets published as a hosted page,
with the demo payload injected at request time.  If no payload has been
exported yet it runs the pipeline on the spot, so ``python -m avr25d.cli
serve`` works on a clean checkout.

Endpoints
    GET /                 the dashboard
    GET /api/data         the full payload (frames + benchmark)
    GET /api/benchmark    the benchmark report alone
    GET /api/config       the pipeline configuration
    GET /healthz          liveness
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(ROOT, "web")
TEMPLATE = os.path.join(WEB_DIR, "dashboard.html")
PAYLOAD = os.path.join(WEB_DIR, "demo_data.json")

SKELETON = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>:root{color-scheme:light dark}body{margin:0}img{max-width:100%%}
[hidden]{display:none!important}</style>
</head><body>
%s
</body></html>"""


def load_payload(frames: int = 6, benchmark_frames: int = 12,
                 checkpoint: Optional[str] = None) -> Dict:
    if os.path.exists(PAYLOAD) and checkpoint is None:
        with open(PAYLOAD) as fh:
            return json.load(fh)
    import sys
    sys.path.insert(0, ROOT)
    from tools.export_demo import build
    payload = build(n_frames=frames, benchmark_frames=benchmark_frames,
                    checkpoint=checkpoint)
    os.makedirs(WEB_DIR, exist_ok=True)
    with open(PAYLOAD, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return payload


def render_page(payload: Dict) -> bytes:
    with open(TEMPLATE) as fh:
        body = fh.read()
    body = body.replace("__AVR_DATA__", json.dumps(payload, separators=(",", ":")))
    return (SKELETON % body).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    payload: Optional[Dict] = None
    page: Optional[bytes] = None

    server_version = "avr25d/1.0"

    def log_message(self, fmt, *args):     # quieter console
        if self.path.startswith("/api") or self.path == "/":
            super().log_message(fmt, *args)

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def do_GET(self):                       # noqa: N802  (stdlib naming)
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(Handler.page, "text/html; charset=utf-8")
        elif path == "/api/data":
            self._json(Handler.payload)
        elif path == "/api/benchmark":
            self._json(Handler.payload["benchmark"])
        elif path == "/api/config":
            self._json(Handler.payload["benchmark"]["config"])
        elif path == "/healthz":
            self._json({"ok": True, "frames": len(Handler.payload["frames"])})
        else:
            self._send(b"not found", "text/plain", 404)


def serve(host: str = "127.0.0.1", port: int = 8080, frames: int = 6,
          benchmark_frames: int = 12, open_browser: bool = False,
          checkpoint: Optional[str] = None) -> None:
    print("[*] preparing demo payload ...")
    Handler.payload = load_payload(frames, benchmark_frames, checkpoint)
    Handler.page = render_page(Handler.payload)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"[+] AVR-2.5D dashboard on {url}")
    print(f"    {len(Handler.payload['frames'])} frames, "
          f"{len(Handler.page) / 1e6:.1f} MB page")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    serve()
