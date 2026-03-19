"""Merge adjacent tracked-change wrappers from the same author.

When multiple sequential edits occur, Word creates separate <w:ins> or <w:del>
elements for each edit even when the author is identical. This produces verbose
XML that is harder to read and edit. This module collapses them into single
wrappers where safe to do so.

Merge criteria:
- Same element type (ins+ins or del+del only)
- Same author attribute (timestamps may differ)
- No intervening element nodes between them
"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import defusedxml.minidom

WML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def merge_adjacent_revisions(workspace: str) -> tuple[int, str]:
    main_doc = Path(workspace) / "word" / "document.xml"
    if not main_doc.exists():
        return 0, f"Error: {main_doc} not found"

    try:
        dom = defusedxml.minidom.parseString(main_doc.read_text(encoding="utf-8"))
        root = dom.documentElement

        total = 0
        for container in _scan_for(root, "p") + _scan_for(root, "tc"):
            total += _collapse_adjacent(container, "ins")
            total += _collapse_adjacent(container, "del")

        main_doc.write_bytes(dom.toxml(encoding="UTF-8"))
        return total, f"Merged {total} adjacent revision blocks"
    except Exception as exc:
        return 0, f"Error: {exc}"


def _collapse_adjacent(parent, tag_name: str) -> int:
    candidates = [
        ch for ch in parent.childNodes
        if ch.nodeType == ch.ELEMENT_NODE and _matches_tag(ch, tag_name)
    ]

    if len(candidates) < 2:
        return 0

    collapsed = 0
    idx = 0
    while idx < len(candidates) - 1:
        current = candidates[idx]
        following = candidates[idx + 1]

        if _safe_to_merge(current, following):
            _transfer_children(current, following)
            parent.removeChild(following)
            candidates.pop(idx + 1)
            collapsed += 1
        else:
            idx += 1

    return collapsed


def _matches_tag(node, local_name: str) -> bool:
    tag = node.localName or node.tagName
    return tag == local_name or tag.endswith(f":{local_name}")


def _extract_author(elem) -> str:
    author = elem.getAttribute("w:author")
    if not author:
        for attr in elem.attributes.values():
            if attr.localName == "author" or attr.name.endswith(":author"):
                return attr.value
    return author


def _safe_to_merge(elem_a, elem_b) -> bool:
    if _extract_author(elem_a) != _extract_author(elem_b):
        return False

    cursor = elem_a.nextSibling
    while cursor and cursor != elem_b:
        if cursor.nodeType == cursor.ELEMENT_NODE:
            return False
        if cursor.nodeType == cursor.TEXT_NODE and cursor.data.strip():
            return False
        cursor = cursor.nextSibling

    return True


def _transfer_children(target, source):
    while source.firstChild:
        child = source.firstChild
        source.removeChild(child)
        target.appendChild(child)


def _scan_for(root, local_name: str) -> list:
    found = []

    def walk(node):
        if node.nodeType == node.ELEMENT_NODE:
            tag = node.localName or node.tagName
            if tag == local_name or tag.endswith(f":{local_name}"):
                found.append(node)
            for ch in node.childNodes:
                walk(ch)

    walk(root)
    return found


def discover_revision_authors(doc_xml_path: Path) -> dict[str, int]:
    if not doc_xml_path.exists():
        return {}

    try:
        tree = ET.parse(doc_xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return {}

    ns = {"w": WML_NS}
    author_key = f"{{{WML_NS}}}author"
    tally: dict[str, int] = {}

    for tag in ("ins", "del"):
        for elem in root.findall(f".//w:{tag}", ns):
            who = elem.get(author_key)
            if who:
                tally[who] = tally.get(who, 0) + 1

    return tally


def _extract_authors_from_archive(docx_path: Path) -> dict[str, int]:
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return {}
            with zf.open("word/document.xml") as fh:
                tree = ET.parse(fh)
                root = tree.getroot()
                ns = {"w": WML_NS}
                author_key = f"{{{WML_NS}}}author"
                tally: dict[str, int] = {}
                for tag in ("ins", "del"):
                    for elem in root.findall(f".//w:{tag}", ns):
                        who = elem.get(author_key)
                        if who:
                            tally[who] = tally.get(who, 0) + 1
                return tally
    except (zipfile.BadZipFile, ET.ParseError):
        return {}


def detect_editing_author(
    modified_dir: Path,
    original_archive: Path,
    fallback: str = "Verdent",
) -> str:
    modified_xml = modified_dir / "word" / "document.xml"
    current_authors = discover_revision_authors(modified_xml)

    if not current_authors:
        return fallback

    baseline_authors = _extract_authors_from_archive(original_archive)

    new_contributions: dict[str, int] = {}
    for author, count in current_authors.items():
        baseline = baseline_authors.get(author, 0)
        delta = count - baseline
        if delta > 0:
            new_contributions[author] = delta

    if not new_contributions:
        return fallback

    if len(new_contributions) == 1:
        return next(iter(new_contributions))

    raise ValueError(
        f"Multiple authors contributed new changes: {new_contributions}. "
        "Cannot determine the primary editing author."
    )
