"""Slide rendering to JPEG images via LibreOffice + Poppler."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .runtime.converter import lo_env

DEFAULT_DPI = 150


def render_slides(
    pptx_path: str,
    page_range: str | None = None,
    dpi: int = DEFAULT_DPI,
    dest: str = ".",
) -> list[str]:
    src = Path(pptx_path)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")

    outdir = Path(dest)
    outdir.mkdir(parents=True, exist_ok=True)

    import shutil
    if not shutil.which("soffice"):
        raise RuntimeError(
            "LibreOffice (soffice) is not installed or not in PATH. "
            "Install it: brew install --cask libreoffice"
        )
    if not shutil.which("pdftoppm"):
        raise RuntimeError(
            "Poppler (pdftoppm) is not installed or not in PATH. "
            "Install it: brew install poppler"
        )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pdf = td / f"{src.stem}.pdf"

        res = subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(td), str(src)],
            capture_output=True, text=True, env=lo_env(),
        )
        if res.returncode != 0 or not pdf.exists():
            raise RuntimeError(f"PDF conversion failed: {res.stderr.strip()}")

        cmd = ["pdftoppm", "-jpeg", "-r", str(dpi)]

        if page_range:
            first, last = _resolve_range(page_range)
            if first is not None:
                cmd.extend(["-f", str(first)])
            if last is not None:
                cmd.extend(["-l", str(last)])

        cmd.extend([str(pdf), str(outdir / "slide")])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Rasterization failed: {res.stderr.strip()}")

    return sorted(str(p) for p in outdir.glob("slide-*.jpg"))


def _resolve_range(spec: str) -> tuple[int | None, int | None]:
    spec = spec.strip()
    if "-" in spec and not spec.startswith("-"):
        a, b = spec.split("-", 1)
        return int(a), int(b)
    elif "," in spec:
        nums = [int(x.strip()) for x in spec.split(",")]
        return min(nums), max(nums)
    else:
        n = int(spec)
        return n, n
