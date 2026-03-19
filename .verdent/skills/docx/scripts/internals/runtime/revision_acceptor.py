"""Accept all tracked changes in a Word document via LibreOffice macro.

Produces a clean document with no revision marks by invoking a Basic
macro through LibreOffice in headless mode.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from .converter import get_office_env

log = logging.getLogger(__name__)

_PROFILE_ROOT = "/tmp/verdent_lo_profile"
_MACRO_HOME = f"{_PROFILE_ROOT}/user/basic/Standard"

_ACCEPT_ALL_MACRO = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" script:language="StarBasic">
    Sub FinalizeRevisions()
        Dim frame As Object
        Dim dispatch As Object

        frame = ThisComponent.CurrentController.Frame
        dispatch = createUnoService("com.sun.star.frame.DispatchHelper")

        dispatch.executeDispatch(frame, ".uno:AcceptAllTrackedChanges", "", 0, Array())
        ThisComponent.store()
        ThisComponent.close(True)
    End Sub
</script:module>"""


def finalize_revisions(
    source_path: str,
    dest_path: str,
) -> tuple[None, str]:
    src = Path(source_path)
    dst = Path(dest_path)

    if not src.exists():
        return None, f"Error: Source file not found: {source_path}"

    if src.suffix.lower() != ".docx":
        return None, f"Error: Expected .docx file, got: {source_path}"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as exc:
        return None, f"Error: Could not prepare output file: {exc}"

    if not _provision_macro():
        return None, "Error: LibreOffice macro setup failed"

    command = [
        "soffice",
        "--headless",
        f"-env:UserInstallation=file://{_PROFILE_ROOT}",
        "--norestore",
        "vnd.sun.star.script:Standard.Module1.FinalizeRevisions?language=Basic&location=application",
        str(dst.absolute()),
    ]

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=get_office_env(),
        )
    except subprocess.TimeoutExpired:
        return None, f"Revisions accepted: {source_path} -> {dest_path}"

    if proc.returncode != 0:
        return None, f"Error: LibreOffice returned: {proc.stderr}"

    return None, f"Revisions accepted: {source_path} -> {dest_path}"


def _provision_macro() -> bool:
    macro_dir = Path(_MACRO_HOME)
    macro_file = macro_dir / "Module1.xba"

    if macro_file.exists() and "FinalizeRevisions" in macro_file.read_text():
        return True

    if not macro_dir.exists():
        subprocess.run(
            ["soffice", "--headless",
             f"-env:UserInstallation=file://{_PROFILE_ROOT}",
             "--terminate_after_init"],
            capture_output=True,
            timeout=10,
            check=False,
            env=get_office_env(),
        )
        macro_dir.mkdir(parents=True, exist_ok=True)

    try:
        macro_file.write_text(_ACCEPT_ALL_MACRO)
        return True
    except Exception as exc:
        log.warning("Macro provisioning failed: %s", exc)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept all tracked changes")
    parser.add_argument("source", help="Input .docx with revisions")
    parser.add_argument("dest", help="Output .docx (clean)")
    args = parser.parse_args()

    _, msg = finalize_revisions(args.source, args.dest)
    print(msg)
    if msg.startswith("Error"):
        raise SystemExit(1)
