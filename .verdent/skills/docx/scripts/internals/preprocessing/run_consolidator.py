"""Consolidate adjacent text runs that share identical formatting.

When Word saves a document, it often splits logically continuous text into
many tiny <w:r> elements due to revision tracking, spell-check, or cursor
position history. This module recombines them for cleaner XML editing.

Also strips rsid attributes (revision session IDs) and proofErr markers
that fragment runs without affecting visible output.
"""

from pathlib import Path

import defusedxml.minidom


def consolidate_adjacent_runs(workspace: str) -> tuple[int, str]:
    main_doc = Path(workspace) / "word" / "document.xml"
    if not main_doc.exists():
        return 0, f"Error: {main_doc} not found"

    try:
        dom = defusedxml.minidom.parseString(main_doc.read_text(encoding="utf-8"))
        root = dom.documentElement

        _purge_nodes_by_tag(root, "proofErr")
        _clear_session_markers(root)

        parents = {r.parentNode for r in _collect_by_tag(root, "r")}
        total = sum(_merge_within(container) for container in parents)

        main_doc.write_bytes(dom.toxml(encoding="UTF-8"))
        return total, f"Consolidated {total} runs"
    except Exception as exc:
        return 0, f"Error: {exc}"


def _collect_by_tag(root, local_name: str) -> list:
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


def _tag_matches(node, local_name: str) -> bool:
    tag = node.localName or node.tagName
    return tag == local_name or tag.endswith(f":{local_name}")


def _get_first_child(parent, local_name: str):
    for ch in parent.childNodes:
        if ch.nodeType == ch.ELEMENT_NODE and _tag_matches(ch, local_name):
            return ch
    return None


def _get_all_children(parent, local_name: str) -> list:
    return [
        ch for ch in parent.childNodes
        if ch.nodeType == ch.ELEMENT_NODE and _tag_matches(ch, local_name)
    ]


def _purge_nodes_by_tag(root, local_name: str):
    for node in _collect_by_tag(root, local_name):
        if node.parentNode:
            node.parentNode.removeChild(node)


def _clear_session_markers(root):
    for run in _collect_by_tag(root, "r"):
        for attr in list(run.attributes.values()):
            if "rsid" in attr.name.lower():
                run.removeAttribute(attr.name)


def _next_element(node):
    sibling = node.nextSibling
    while sibling:
        if sibling.nodeType == sibling.ELEMENT_NODE:
            return sibling
        sibling = sibling.nextSibling
    return None


def _next_run_sibling(node):
    sibling = node.nextSibling
    while sibling:
        if sibling.nodeType == sibling.ELEMENT_NODE:
            if _tag_matches(sibling, "r"):
                return sibling
            return None
        if sibling.nodeType == sibling.TEXT_NODE and sibling.data.strip():
            return None
        sibling = sibling.nextSibling
    return None


def _formatting_matches(run_a, run_b) -> bool:
    props_a = _get_first_child(run_a, "rPr")
    props_b = _get_first_child(run_b, "rPr")

    if (props_a is None) != (props_b is None):
        return False
    if props_a is None:
        return True
    return props_a.toxml() == props_b.toxml()


def _absorb_content(target, source):
    for child in list(source.childNodes):
        if child.nodeType == child.ELEMENT_NODE:
            tag = child.localName or child.tagName
            if not (tag == "rPr" or tag.endswith(":rPr")):
                target.appendChild(child)


def _unify_text_nodes(run):
    text_elems = _get_all_children(run, "t")
    for i in range(len(text_elems) - 1, 0, -1):
        curr = text_elems[i]
        prev = text_elems[i - 1]

        prev_val = prev.firstChild.data if prev.firstChild else ""
        curr_val = curr.firstChild.data if curr.firstChild else ""
        combined = prev_val + curr_val

        if prev.firstChild:
            prev.firstChild.data = combined
        else:
            prev.appendChild(run.ownerDocument.createTextNode(combined))

        if combined.startswith(" ") or combined.endswith(" "):
            prev.setAttribute("xml:space", "preserve")
        elif prev.hasAttribute("xml:space"):
            prev.removeAttribute("xml:space")

        run.removeChild(curr)


def _merge_within(container) -> int:
    merged = 0
    run = None
    for ch in container.childNodes:
        if ch.nodeType == ch.ELEMENT_NODE and _tag_matches(ch, "r"):
            run = ch
            break

    if not run:
        return 0

    while run:
        while True:
            neighbor = _next_element(run)
            if neighbor and _tag_matches(neighbor, "r") and _formatting_matches(run, neighbor):
                _absorb_content(run, neighbor)
                container.removeChild(neighbor)
                merged += 1
            else:
                break

        _unify_text_nodes(run)

        nxt = run.nextSibling
        run = None
        while nxt:
            if nxt.nodeType == nxt.ELEMENT_NODE and _tag_matches(nxt, "r"):
                run = nxt
                break
            nxt = nxt.nextSibling

    return merged
