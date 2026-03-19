"""PPTX archive disassembly and reassembly."""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import defusedxml.minidom

_SMART_QUOTES = {
    "\u201c": "&#x201C;",
    "\u201d": "&#x201D;",
    "\u2018": "&#x2018;",
    "\u2019": "&#x2019;",
}


def disassemble(src: str, dest: str) -> str:
    src_p = Path(src)
    dest_p = Path(dest)

    if not src_p.exists():
        raise FileNotFoundError(f"{src} does not exist")
    if src_p.suffix.lower() not in (".pptx", ".docx", ".xlsx"):
        raise ValueError(f"Unsupported file type: {src_p.suffix}")

    try:
        dest_p.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(src_p, "r") as zf:
            zf.extractall(dest_p)

        xml_files = list(dest_p.rglob("*.xml")) + list(dest_p.rglob("*.rels"))

        for xf in xml_files:
            _prettify(xf)

        for xf in xml_files:
            _escape_quotes(xf)

        return f"Extracted {src} -> {dest} ({len(xml_files)} XML files formatted)"

    except zipfile.BadZipFile:
        raise ValueError(f"{src} is not a valid ZIP/Office file")


def reassemble(
    workspace: str,
    output: str,
    reference: str | None = None,
    validate: bool = True,
) -> str:
    ws = Path(workspace)
    out = Path(output)

    if not ws.is_dir():
        raise NotADirectoryError(f"{workspace} is not a directory")

    if validate and reference:
        ref = Path(reference)
        if ref.exists():
            from .integrity.checks import DeckIntegrityChecker
            checker = DeckIntegrityChecker(ws, ref)
            fixes = checker.auto_repair()
            if fixes:
                print(f"Auto-repaired {fixes} issue(s)")
            if not checker.run_all():
                raise RuntimeError("Integrity checks failed — see above")

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "staged"
        shutil.copytree(ws, staging)

        for pat in ("*.xml", "*.rels"):
            for xf in staging.rglob(pat):
                _minify_xml(xf)

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in staging.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(staging))

    return f"Packed {workspace} -> {output}"


def _prettify(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
        dom = defusedxml.minidom.parseString(raw)
        path.write_bytes(dom.toprettyxml(indent="  ", encoding="utf-8"))
    except Exception:
        pass


def _escape_quotes(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        for ch, ent in _SMART_QUOTES.items():
            text = text.replace(ch, ent)
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _minify_xml(path: Path) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            dom = defusedxml.minidom.parse(f)

        for el in dom.getElementsByTagName("*"):
            if el.tagName.endswith(":t"):
                continue
            for child in list(el.childNodes):
                if (
                    child.nodeType == child.TEXT_NODE
                    and child.nodeValue
                    and child.nodeValue.strip() == ""
                ) or child.nodeType == child.COMMENT_NODE:
                    el.removeChild(child)

        path.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception as e:
        print(f"Warning: Could not compress {path.name}: {e}", file=sys.stderr)
        raise
