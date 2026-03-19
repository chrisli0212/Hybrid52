"""Orchestrate all document integrity checks.

Entry point for the `verify` subcommand. Routes to the appropriate
checker classes based on file type.
"""

import sys
import tempfile
import zipfile
from pathlib import Path


def run_checks(
    target: str,
    original: str | None = None,
    auto_fix: bool = False,
    author: str = "Verdent",
    verbose: bool = False,
) -> bool:
    path = Path(target)
    assert path.exists(), f"{path} does not exist"

    original_path = Path(original) if original else None
    if original_path:
        assert original_path.is_file(), f"{original_path} is not a file"

    file_ext = (original_path or path).suffix.lower()
    assert file_ext in {".docx", ".pptx", ".xlsx"}, (
        f"Cannot determine type from {path}. Provide --original or use a .docx/.pptx/.xlsx file."
    )

    if path.is_file() and path.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
        work_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(work_dir)
        unpacked = Path(work_dir)
    else:
        assert path.is_dir(), f"{path} is not a directory or Office file"
        unpacked = path

    if file_ext == ".docx":
        from .schema_checker import WordDocumentChecker
        from .revision_checker import RevisionIntegrityChecker

        checkers = [WordDocumentChecker(unpacked, original_path, verbose=verbose)]
        if original_path:
            checkers.append(
                RevisionIntegrityChecker(
                    unpacked, original_path, verbose=verbose, author=author
                )
            )
    else:
        print(f"Validation not yet supported for {file_ext}")
        sys.exit(1)

    if auto_fix:
        total_fixes = sum(c.auto_repair() for c in checkers)
        if total_fixes:
            print(f"Auto-repaired {total_fixes} issue(s)")

    passed = all(c.run_all() for c in checkers)

    if passed:
        print("All checks passed")

    return passed
