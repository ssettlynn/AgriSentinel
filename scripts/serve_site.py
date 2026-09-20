"""
scripts/serve_site.py — serve the AgriSentinel static web UI locally.

Run this file straight from PyCharm (right-click -> Run 'serve_site'), or from a
terminal: `python scripts/serve_site.py`. It starts a small HTTP server for the
site/ folder and opens your browser at http://localhost:8000.

Why a server (not just double-clicking index.html): opening via file:// works for
most pages, but a real http:// server matches how the site is actually deployed
(GitHub Pages) and avoids any browser file-access quirks. No Flask/Django needed —
this uses Python's built-in http.server.
"""
from __future__ import annotations

import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

PORT = 8000
SITE_DIR = Path(__file__).resolve().parent.parent / "site"   # AgriSentinel/site


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve files out of site/ (not the current working directory)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def log_message(self, fmt, *args):   # quieter console
        print("  ", self.address_string(), "-", fmt % args)


def main() -> None:
    if not (SITE_DIR / "index.html").exists():
        raise SystemExit(f"index.html not found in {SITE_DIR} — is the site/ folder present?")

    data_js = SITE_DIR / "agrisentinel_data.js"
    if not data_js.exists():
        print("WARNING: site/agrisentinel_data.js is missing — the pages will load")
        print("         but show no data. Generate it first with:")
        print("             python scripts/run_pipeline.py --stage export")
        print()

    url = f"http://localhost:{PORT}/index.html"
    # open the browser a moment after the server starts listening
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("=" * 60)
        print(f"  AgriSentinel UI running at:  {url}")
        print(f"  Serving folder:              {SITE_DIR}")
        print("  Press Ctrl+C (or the red Stop button in PyCharm) to stop.")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nserver stopped.")


if __name__ == "__main__":
    main()
