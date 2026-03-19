"""Render Word document pages as JPEG images for visual inspection.

Uses LibreOffice for PDF conversion and Poppler for rasterization.
"""

import subprocess
import tempfile
from pathlib import Path

from .converter import get_office_env


def pages_to_images(
    doc_path: str,
    page_range: str | None = None,
    resolution: int = 150,
    output_dir: str | None = None,
) -> list[Path]:
    source = Path(doc_path)
    if not source.exists():
        raise FileNotFoundError(f"{doc_path} not found")

    dest = Path(output_dir) if output_dir else source.parent
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)

        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(staging_path), str(source)],
            env=get_office_env(),
            capture_output=True,
            timeout=60,
            check=False,
        )

        pdf_candidates = list(staging_path.glob("*.pdf"))
        if not pdf_candidates:
            raise RuntimeError("PDF conversion failed — no output produced")

        pdf_file = pdf_candidates[0]

        poppler_args = [
            "pdftoppm", "-jpeg", "-r", str(resolution),
        ]
        if page_range:
            parts = page_range.split("-")
            poppler_args.extend(["-f", parts[0]])
            if len(parts) > 1:
                poppler_args.extend(["-l", parts[1]])

        poppler_args.extend([str(pdf_file), str(dest / "page")])

        subprocess.run(poppler_args, capture_output=True, timeout=120, check=True)

    outputs = sorted(dest.glob("page-*.jpg"))
    if outputs:
        print(f"Rendered {len(outputs)} page(s) to {dest}")
    return outputs
