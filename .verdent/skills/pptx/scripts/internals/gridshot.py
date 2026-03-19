"""Thumbnail grid generation for slide layout analysis."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import defusedxml.minidom
from PIL import Image, ImageDraw, ImageFont

from .runtime.converter import lo_env

CELL_WIDTH = 300
RENDER_DPI = 100
MAX_COLS = 6
JPEG_QUALITY = 95
GUTTER = 20
BORDER_PX = 2
LABEL_SIZE_RATIO = 0.10


def generate_grid(pptx_path: str, columns: int = 3) -> list[str]:
    src = Path(pptx_path)
    cols = min(columns, MAX_COLS)

    if not src.exists() or src.suffix.lower() != ".pptx":
        raise ValueError(f"Not a valid .pptx file: {src}")

    slide_info = _parse_slide_list(src)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raster_imgs = _slides_to_images(src, td)

        if not raster_imgs and not any(s["hidden"] for s in slide_info):
            raise RuntimeError("No slides found in presentation")

        cells = _assemble_cells(slide_info, raster_imgs, td)
        return _render_grids(cells, cols, src.stem)


def _parse_slide_list(pptx: Path) -> list[dict]:
    with zipfile.ZipFile(pptx, "r") as zf:
        rels_xml = zf.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
        rels_dom = defusedxml.minidom.parseString(rels_xml)
        targets = {}
        for r in rels_dom.getElementsByTagName("Relationship"):
            if "slide" in r.getAttribute("Type") and r.getAttribute("Target").startswith("slides/"):
                targets[r.getAttribute("Id")] = r.getAttribute("Target").rsplit("/", 1)[-1]

        pres_xml = zf.read("ppt/presentation.xml").decode("utf-8")
        pres_dom = defusedxml.minidom.parseString(pres_xml)

        entries = []
        for node in pres_dom.getElementsByTagName("p:sldId"):
            rid = node.getAttribute("r:id")
            if rid in targets:
                entries.append({
                    "filename": targets[rid],
                    "hidden": node.getAttribute("show") == "0",
                })
    return entries


def _slides_to_images(pptx: Path, workdir: Path) -> list[Path]:
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

    pdf = workdir / f"{pptx.stem}.pdf"
    res = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(workdir), str(pptx)],
        capture_output=True, text=True, env=lo_env(),
    )
    if not pdf.exists():
        raise RuntimeError(f"PDF conversion failed: {res.stderr.strip()}")

    res = subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(RENDER_DPI), str(pdf), str(workdir / "pg")],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {res.stderr.strip()}")
    return sorted(workdir.glob("pg-*.jpg"))


def _assemble_cells(info: list[dict], images: list[Path], workdir: Path) -> list[tuple[Path, str]]:
    ref_size = Image.open(images[0]).size if images else (1920, 1080)
    cells = []
    vis_idx = 0
    for s in info:
        if s["hidden"]:
            ph = workdir / f"ph-{s['filename']}.jpg"
            _make_hatch(ref_size).save(ph, "JPEG")
            cells.append((ph, f"{s['filename']} (hidden)"))
        else:
            if vis_idx < len(images):
                cells.append((images[vis_idx], s["filename"]))
                vis_idx += 1
    return cells


def _make_hatch(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, "#F0F0F0")
    draw = ImageDraw.Draw(img)
    pen = max(5, min(size) // 100)
    draw.line([(0, 0), size], fill="#CCCCCC", width=pen)
    draw.line([(size[0], 0), (0, size[1])], fill="#CCCCCC", width=pen)
    return img


def _render_grids(cells: list[tuple[Path, str]], cols: int, prefix: str) -> list[str]:
    batch_size = cols * (cols + 1)
    outputs = []

    for chunk_idx, start in enumerate(range(0, len(cells), batch_size)):
        chunk = cells[start:start + batch_size]
        canvas = _paint_grid(chunk, cols)

        suffix = f"-{chunk_idx + 1}" if len(cells) > batch_size else ""
        dest = Path(f"thumbnails{suffix}.jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(dest), quality=JPEG_QUALITY)
        outputs.append(str(dest))

    return outputs


def _paint_grid(cells: list[tuple[Path, str]], cols: int) -> Image.Image:
    fsize = int(CELL_WIDTH * LABEL_SIZE_RATIO)
    label_margin = int(fsize * 0.4)

    with Image.open(cells[0][0]) as sample:
        aspect = sample.height / sample.width
    cell_h = int(CELL_WIDTH * aspect)

    rows = (len(cells) + cols - 1) // cols
    total_w = cols * CELL_WIDTH + (cols + 1) * GUTTER
    total_h = rows * (cell_h + fsize + label_margin * 2) + (rows + 1) * GUTTER

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default(size=fsize)
    except Exception:
        font = ImageFont.load_default()

    for i, (img_path, label) in enumerate(cells):
        row, col = divmod(i, cols)
        x0 = col * CELL_WIDTH + (col + 1) * GUTTER
        y0 = row * (cell_h + fsize + label_margin * 2) + (row + 1) * GUTTER

        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x0 + (CELL_WIDTH - tw) // 2, y0 + label_margin), label, fill="black", font=font)

        img_y = y0 + label_margin + fsize + label_margin
        with Image.open(img_path) as img:
            img.thumbnail((CELL_WIDTH, cell_h), Image.Resampling.LANCZOS)
            iw, ih = img.size
            px = x0 + (CELL_WIDTH - iw) // 2
            py = img_y + (cell_h - ih) // 2
            canvas.paste(img, (px, py))
            if BORDER_PX > 0:
                draw.rectangle(
                    [(px - BORDER_PX, py - BORDER_PX), (px + iw + BORDER_PX - 1, py + ih + BORDER_PX - 1)],
                    outline="gray", width=BORDER_PX,
                )

    return canvas
