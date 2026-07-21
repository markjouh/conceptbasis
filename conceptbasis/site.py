"""Shared primitives for the self-contained public inspection pages."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


PUBLIC_PAGES = (
    ("index", "index.html", "⌂ Concept Basis"),
    ("playground", "playground.html", "Playground"),
    ("baseline", "playground-baseline.html", "Baseline"),
    ("dictionary", "dictionary.html", "Dictionary"),
    ("fixed-labels", "fixed-labels.html", "Fixed labels"),
    ("attributes", "attributes.html", "Open tags"),
)

NAV_STYLE = """<style>
#sitenav{position:fixed;top:10px;right:14px;z-index:99;background:rgba(18,22,28,.94);border:1px solid #38404c;border-radius:20px;padding:6px 14px;font:12px system-ui;display:flex;gap:14px}
#sitenav a{color:#9ab8d8;text-decoration:none}#sitenav a:hover{color:#fff}#sitenav a.here{color:#fff;font-weight:650}
@media(max-width:850px){#sitenav{position:static;margin:8px;overflow:auto;white-space:nowrap}}
</style>"""


def public_nav(active: str) -> str:
    """Render the common public-site navigation with one active page."""
    keys = {key for key, _href, _label in PUBLIC_PAGES}
    if active not in keys:
        raise ValueError(f"unknown public page: {active}")
    links = "".join(
        f'<a href="{href}" class="here">{label}</a>'
        if key == active
        else f'<a href="{href}">{label}</a>'
        for key, href, label in PUBLIC_PAGES
    )
    return f'<div id="sitenav">{links}</div>\n{NAV_STYLE}'


def thumbnail_data_url(
    path: str | Path,
    *,
    size: int,
    quality: int = 76,
) -> str:
    """Encode a bounded WebP thumbnail for a self-contained HTML page."""
    if size < 1 or not 1 <= quality <= 100:
        raise ValueError("size and quality must be positive")
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "WEBP", quality=quality, method=6)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/webp;base64,{encoded}"
