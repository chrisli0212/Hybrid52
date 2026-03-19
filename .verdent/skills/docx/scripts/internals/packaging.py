"""Archive decomposition and composition for Office documents.

Handles unpacking .docx/.pptx/.xlsx into editable directories and
reassembling them back into valid Office files.
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import defusedxml.minidom


TYPOGRAPHIC_ENTITIES = {
    "\u201c": "&#x201C;",
    "\u201d": "&#x201D;",
    "\u2018": "&#x2018;",
    "\u2019": "&#x2019;",
}


def decompose_archive(
    source_path: str,
    dest_dir: str,
    consolidate_runs: bool = True,
    consolidate_revisions: bool = True,
) -> tuple[None, str]:
    src = Path(source_path)
    dst = Path(dest_dir)
    ext = src.suffix.lower()

    if not src.exists():
        return None, f"Error: {source_path} does not exist"

    if ext not in {".docx", ".pptx", ".xlsx"}:
        return None, f"Error: Unsupported format {ext}. Expected .docx, .pptx, or .xlsx"

    try:
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as archive:
            archive.extractall(dst)

        markup_files = list(dst.rglob("*.xml")) + list(dst.rglob("*.rels"))
        for mf in markup_files:
            _format_markup(mf)

        summary = f"Extracted {source_path} ({len(markup_files)} XML files)"

        if ext == ".docx":
            if consolidate_revisions:
                from .preprocessing.revision_merger import merge_adjacent_revisions
                count, _ = merge_adjacent_revisions(str(dst))
                summary += f", consolidated {count} revisions"

            if consolidate_runs:
                from .preprocessing.run_consolidator import consolidate_adjacent_runs
                count, _ = consolidate_adjacent_runs(str(dst))
                summary += f", merged {count} runs"

        for mf in markup_files:
            _encode_typographic_chars(mf)

        return None, summary

    except zipfile.BadZipFile:
        return None, f"Error: {source_path} is not a valid archive"
    except Exception as exc:
        return None, f"Error during extraction: {exc}"


def compose_archive(
    workspace: str,
    output_path: str,
    reference_file: str | None = None,
    run_verification: bool = True,
    author_detector=None,
) -> tuple[None, str]:
    ws = Path(workspace)
    out = Path(output_path)
    ext = out.suffix.lower()

    if not ws.is_dir():
        return None, f"Error: {workspace} is not a directory"

    if ext not in {".docx", ".pptx", ".xlsx"}:
        return None, f"Error: Output must be .docx, .pptx, or .xlsx"

    if run_verification and reference_file:
        ref = Path(reference_file)
        if ref.exists():
            ok, report = _perform_verification(ws, ref, ext, author_detector)
            if report:
                print(report)
            if not ok:
                return None, f"Error: Verification failed for {workspace}"

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging) / "staged"
        shutil.copytree(ws, staging_dir)

        for pattern in ("*.xml", "*.rels"):
            for mf in staging_dir.rglob(pattern):
                _strip_formatting(mf)

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in staging_dir.rglob("*"):
                if item.is_file():
                    archive.write(item, item.relative_to(staging_dir))

    return None, f"Assembled {workspace} -> {output_path}"


def _perform_verification(
    workspace: Path,
    reference: Path,
    ext: str,
    author_detector=None,
) -> tuple[bool, str | None]:
    messages = []

    if ext == ".docx":
        from .integrity.schema_checker import WordDocumentChecker
        from .integrity.revision_checker import RevisionIntegrityChecker

        author = "Verdent"
        if author_detector:
            try:
                author = author_detector(workspace, reference)
            except ValueError as e:
                print(f"Warning: {e} Falling back to 'Verdent'.", file=sys.stderr)

        checkers = [
            WordDocumentChecker(workspace, reference),
            RevisionIntegrityChecker(workspace, reference, author=author),
        ]
    else:
        return True, None

    total_fixes = sum(c.auto_repair() for c in checkers)
    if total_fixes:
        messages.append(f"Auto-repaired {total_fixes} issue(s)")

    passed = all(c.run_all() for c in checkers)
    if passed:
        messages.append("All checks passed")

    return passed, "\n".join(messages) if messages else None


def _format_markup(filepath: Path) -> None:
    try:
        raw = filepath.read_text(encoding="utf-8")
        dom = defusedxml.minidom.parseString(raw)
        filepath.write_bytes(dom.toprettyxml(indent="  ", encoding="utf-8"))
    except Exception:
        pass


def _strip_formatting(filepath: Path) -> None:
    try:
        with open(filepath, encoding="utf-8") as fh:
            dom = defusedxml.minidom.parse(fh)

        for node in dom.getElementsByTagName("*"):
            if node.tagName.endswith(":t"):
                continue
            for child in list(node.childNodes):
                is_blank_text = (
                    child.nodeType == child.TEXT_NODE
                    and child.nodeValue
                    and child.nodeValue.strip() == ""
                )
                is_comment = child.nodeType == child.COMMENT_NODE
                if is_blank_text or is_comment:
                    node.removeChild(child)

        filepath.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception as exc:
        print(f"WARNING: Could not condense {filepath.name}: {exc}", file=sys.stderr)
        raise


def _encode_typographic_chars(filepath: Path) -> None:
    try:
        text = filepath.read_text(encoding="utf-8")
        for char, entity in TYPOGRAPHIC_ENTITIES.items():
            text = text.replace(char, entity)
        filepath.write_text(text, encoding="utf-8")
    except Exception:
        pass
