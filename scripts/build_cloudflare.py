#!/usr/bin/env python3
"""Build the browser-only Cloudflare static asset bundle."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
DIST = ROOT / "dist"


def main() -> None:
    source = APP.read_text()
    match = re.search(r'INDEX_HTML = r"""(.*)"""\n\n\n@app\.get', source, re.S)
    if not match:
        raise SystemExit("Could not find INDEX_HTML in app.py")

    html = match.group(1)
    html = html.replace(
        '<script type="module">',
        '<script>globalThis.__HUSH_BROWSER_ONLY__ = true;</script>\n<script type="module">',
        1,
    )

    DIST.mkdir(exist_ok=True)
    (DIST / "index.html").write_text(html)
    (DIST / "_headers").write_text(
        "/*\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: no-referrer\n"
        "  Permissions-Policy: camera=(), microphone=(), geolocation=()\n"
    )


if __name__ == "__main__":
    main()
