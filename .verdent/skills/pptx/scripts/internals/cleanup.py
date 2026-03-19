"""Orphaned file cleanup for unpacked PPTX workspaces."""
from __future__ import annotations

import re
from pathlib import Path

import defusedxml.minidom


def sweep_workspace(workspace: str) -> list[str]:
    root = Path(workspace)
    if not root.exists():
        raise FileNotFoundError(f"{workspace} not found")

    total_removed = []

    total_removed.extend(_remove_stale_slides(root))
    total_removed.extend(_remove_trash_dir(root))

    while True:
        batch = _remove_dangling_rels(root)
        refs = _collect_all_references(root)
        batch += _remove_unreferenced_files(root, refs)
        if not batch:
            break
        total_removed.extend(batch)

    if total_removed:
        _clean_content_types(root, total_removed)

    return total_removed


def _active_slide_set(root: Path) -> set[str]:
    pres = root / "ppt" / "presentation.xml"
    rels = root / "ppt" / "_rels" / "presentation.xml.rels"
    if not pres.exists() or not rels.exists():
        return set()

    dom = defusedxml.minidom.parse(str(rels))
    rid_to_file = {}
    for r in dom.getElementsByTagName("Relationship"):
        t = r.getAttribute("Target")
        if "slide" in r.getAttribute("Type") and t.startswith("slides/"):
            rid_to_file[r.getAttribute("Id")] = t.rsplit("/", 1)[-1]

    content = pres.read_text(encoding="utf-8")
    live_rids = set(re.findall(r'<p:sldId[^>]*r:id="([^"]+)"', content))
    return {rid_to_file[r] for r in live_rids if r in rid_to_file}


def _remove_stale_slides(root: Path) -> list[str]:
    slides_dir = root / "ppt" / "slides"
    rels_dir = slides_dir / "_rels"
    pres_rels = root / "ppt" / "_rels" / "presentation.xml.rels"

    if not slides_dir.exists():
        return []

    live = _active_slide_set(root)
    removed = []

    for f in slides_dir.glob("slide*.xml"):
        if f.name not in live:
            rp = str(f.relative_to(root))
            f.unlink()
            removed.append(rp)
            rf = rels_dir / f"{f.name}.rels"
            if rf.exists():
                rf.unlink()
                removed.append(str(rf.relative_to(root)))

    if removed and pres_rels.exists():
        dom = defusedxml.minidom.parse(str(pres_rels))
        changed = False
        for r in list(dom.getElementsByTagName("Relationship")):
            t = r.getAttribute("Target")
            if t.startswith("slides/") and t.rsplit("/", 1)[-1] not in live:
                r.parentNode.removeChild(r)
                changed = True
        if changed:
            pres_rels.write_bytes(dom.toxml(encoding="utf-8"))

    return removed


def _remove_trash_dir(root: Path) -> list[str]:
    trash = root / "[trash]"
    removed = []
    if trash.exists() and trash.is_dir():
        for f in trash.iterdir():
            if f.is_file():
                removed.append(str(f.relative_to(root)))
                f.unlink()
        trash.rmdir()
    return removed


def _collect_all_references(root: Path) -> set:
    refs = set()
    for rf in root.rglob("*.rels"):
        dom = defusedxml.minidom.parse(str(rf))
        for r in dom.getElementsByTagName("Relationship"):
            target = r.getAttribute("Target")
            if not target:
                continue
            resolved = (rf.parent.parent / target).resolve()
            try:
                refs.add(resolved.relative_to(root.resolve()))
            except ValueError:
                pass
    return refs


def _slide_level_refs(root: Path) -> set:
    refs = set()
    rels_dir = root / "ppt" / "slides" / "_rels"
    if not rels_dir.exists():
        return refs
    for rf in rels_dir.glob("*.rels"):
        dom = defusedxml.minidom.parse(str(rf))
        for r in dom.getElementsByTagName("Relationship"):
            target = r.getAttribute("Target")
            if not target:
                continue
            resolved = (rf.parent.parent / target).resolve()
            try:
                refs.add(resolved.relative_to(root.resolve()))
            except ValueError:
                pass
    return refs


def _remove_dangling_rels(root: Path) -> list[str]:
    removed = []
    sld_refs = _slide_level_refs(root)
    for dirname in ("charts", "diagrams", "drawings"):
        rdir = root / "ppt" / dirname / "_rels"
        if not rdir.exists():
            continue
        for rf in rdir.glob("*.rels"):
            resource = rdir.parent / rf.name.replace(".rels", "")
            try:
                rel_path = resource.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            if not resource.exists() or rel_path not in sld_refs:
                rf.unlink()
                removed.append(str(rf.relative_to(root)))
    return removed


def _remove_unreferenced_files(root: Path, referenced: set) -> list[str]:
    removed = []

    for dirname in ("media", "embeddings", "charts", "diagrams", "tags", "drawings", "ink"):
        d = root / "ppt" / dirname
        if not d.exists():
            continue
        for f in d.glob("*"):
            if f.is_file() and f.relative_to(root) not in referenced:
                f.unlink()
                removed.append(str(f.relative_to(root)))

    notes_dir = root / "ppt" / "notesSlides"
    if notes_dir.exists():
        for f in notes_dir.glob("*.xml"):
            if f.is_file() and f.relative_to(root) not in referenced:
                f.unlink()
                removed.append(str(f.relative_to(root)))
        nrels = notes_dir / "_rels"
        if nrels.exists():
            for rf in nrels.glob("*.rels"):
                nf = notes_dir / rf.name.replace(".rels", "")
                if not nf.exists():
                    rf.unlink()
                    removed.append(str(rf.relative_to(root)))

    return removed


def _clean_content_types(root: Path, removed: list[str]) -> None:
    ct = root / "[Content_Types].xml"
    if not ct.exists():
        return
    dom = defusedxml.minidom.parse(str(ct))
    changed = False
    for ov in list(dom.getElementsByTagName("Override")):
        pn = ov.getAttribute("PartName").lstrip("/")
        if pn in removed:
            ov.parentNode.removeChild(ov)
            changed = True
    if changed:
        ct.write_bytes(dom.toxml(encoding="utf-8"))
