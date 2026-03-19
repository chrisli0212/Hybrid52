"""Slide structural operations — clone existing slides or instantiate from layouts."""
from __future__ import annotations

import re
import shutil
from pathlib import Path


_EMPTY_SLIDE = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>'''

_LAYOUT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/{layout}"/>
</Relationships>'''


def clone_or_instantiate(workspace: str, source: str) -> None:
    root = Path(workspace)
    if not root.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    if source.startswith("slideLayout") and source.endswith(".xml"):
        _from_layout(root, source)
    elif source.endswith(".xml"):
        _duplicate(root, source)
    else:
        raise ValueError(f"Source must be slide*.xml or slideLayout*.xml, got: {source}")


def _from_layout(root: Path, layout_name: str) -> None:
    layout_path = root / "ppt" / "slideLayouts" / layout_name
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout not found: {layout_path}")

    slides_dir = root / "ppt" / "slides"
    num = _next_num(slides_dir)
    fname = f"slide{num}.xml"

    (slides_dir / fname).write_text(_EMPTY_SLIDE, encoding="utf-8")

    rels_dir = slides_dir / "_rels"
    rels_dir.mkdir(exist_ok=True)
    (rels_dir / f"{fname}.rels").write_text(
        _LAYOUT_RELS.format(layout=layout_name), encoding="utf-8"
    )

    _register_ct(root, fname)
    rid = _register_pres_rel(root, fname)
    sid = _next_sid(root)

    print(f"Created {fname} from layout {layout_name}")
    print(f'Add to <p:sldIdLst>: <p:sldId id="{sid}" r:id="{rid}"/>')


def _duplicate(root: Path, source_name: str) -> None:
    slides_dir = root / "ppt" / "slides"
    rels_dir = slides_dir / "_rels"
    source = slides_dir / source_name

    if not source.exists():
        raise FileNotFoundError(f"Slide not found: {source}")

    num = _next_num(slides_dir)
    fname = f"slide{num}.xml"

    shutil.copy2(source, slides_dir / fname)

    src_rels = rels_dir / f"{source_name}.rels"
    dst_rels = rels_dir / f"{fname}.rels"
    if src_rels.exists():
        shutil.copy2(src_rels, dst_rels)
        rels_text = dst_rels.read_text(encoding="utf-8")
        rels_text = re.sub(
            r'\s*<Relationship[^>]*Type="[^"]*notesSlide"[^>]*/>\s*',
            "\n", rels_text,
        )
        dst_rels.write_text(rels_text, encoding="utf-8")

    _register_ct(root, fname)
    rid = _register_pres_rel(root, fname)
    sid = _next_sid(root)

    print(f"Cloned {source_name} -> {fname}")
    print(f'Add to <p:sldIdLst>: <p:sldId id="{sid}" r:id="{rid}"/>')


def _next_num(slides_dir: Path) -> int:
    nums = [
        int(m.group(1))
        for f in slides_dir.glob("slide*.xml")
        if (m := re.match(r"slide(\d+)\.xml", f.name))
    ]
    return max(nums, default=0) + 1


def _next_sid(root: Path) -> int:
    pres = (root / "ppt" / "presentation.xml").read_text(encoding="utf-8")
    ids = [int(x) for x in re.findall(r'<p:sldId[^>]*id="(\d+)"', pres)]
    base = max(ids, default=255) + 1

    tracker = root / ".verdent_sid_counter"
    if tracker.exists():
        prev = int(tracker.read_text().strip())
        base = max(base, prev + 1)
    tracker.write_text(str(base))
    return base


def _register_ct(root: Path, fname: str) -> None:
    ct_path = root / "[Content_Types].xml"
    ct = ct_path.read_text(encoding="utf-8")
    part_name = f"/ppt/slides/{fname}"
    if part_name not in ct:
        entry = (
            f'<Override PartName="{part_name}" '
            f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
        ct = ct.replace("</Types>", f"  {entry}\n</Types>")
        ct_path.write_text(ct, encoding="utf-8")


def _register_pres_rel(root: Path, fname: str) -> str:
    rels_path = root / "ppt" / "_rels" / "presentation.xml.rels"
    rels = rels_path.read_text(encoding="utf-8")

    existing_ids = [int(x) for x in re.findall(r'Id="rId(\d+)"', rels)]
    rid = f"rId{max(existing_ids, default=0) + 1}"

    if f"slides/{fname}" not in rels:
        tag = (
            f'<Relationship Id="{rid}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/{fname}"/>'
        )
        rels = rels.replace("</Relationships>", f"  {tag}\n</Relationships>")
        rels_path.write_text(rels, encoding="utf-8")

    return rid
