"""Ledgerwise web server (standard library only).  Run:  python server.py [port]"""
import io
import json
import os
import re
import sys
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parent


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from .env so keys never need to be typed in the UI or terminal."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("\"'")


load_env()
from pydantic import BaseModel  # noqa: E402

from graph import LLMClient, stream_audit  # noqa: E402

PAGE = ROOT / "web" / "index.html"


def ser(o):
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, dict):
        return {k: ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [ser(v) for v in o]
    return str(o) if isinstance(o, date) else o


def detect_vendor(raw: str):
    m = (re.search(r"^Vendor:.*\(([A-Za-z0-9_\- ]+)\)\s*$", raw, re.M | re.I)
         or re.search(r"^Vendor ID:\s*(\S+)", raw, re.M | re.I))
    return m.group(1).strip() if m else None


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), code=code)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._json({"ai": LLMClient().available})
        elif path == "/favicon.ico":
            self._send(b"", "image/x-icon", 204)
        else:
            self._send(b"Not found", "text/plain", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 10_000_000:
            return self._json({"error": "File is larger than 10 MB."}, 413)
        data = self.rfile.read(n)
        if self.path == "/api/extract":
            try:
                text = pdf_text(data)
            except ImportError:
                return self._json({"error": "PDF support needs pypdf. Run: pip install pypdf"}, 400)
            except Exception:
                return self._json({"error": "Could not read this PDF."}, 400)
            if not text.strip():
                return self._json({"error": "No selectable text found. Scanned PDFs are not supported yet."}, 400)
            return self._json({"text": text})
        if self.path != "/api/audit":
            return self._send(b"Not found", "text/plain", 404)

        raw = json.loads(data or b"{}").get("raw", "")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(ser(obj)) + "\n").encode())
            self.wfile.flush()

        try:
            vendor = detect_vendor(raw)
            if not vendor:
                return emit({"error": "Could not find a vendor ID in this invoice. "
                                      "Expected a line like: Vendor: Acme Software Solutions LLC (VENDOR-A)"})
            for node, upd in stream_audit(raw, vendor, LLMClient()):
                emit({"node": node, "update": upd})
            emit({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            emit({"error": f"{e}" if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    llm = LLMClient()
    print(f"AI extraction: {llm.provider} ({llm.model})" if llm.available
          else "AI extraction: NOT configured (add a key to .env). Using the built-in parser.")
    print(f"Ledgerwise running at http://localhost:{port}  (Ctrl+C to stop)")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    webbrowser.open(f"http://localhost:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
