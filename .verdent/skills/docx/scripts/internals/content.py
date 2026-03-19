"""Insert annotations (comments) into unpacked Word documents.

Manages the multi-file comment structure: comments.xml, commentsExtended.xml,
commentsIds.xml, commentsExtensible.xml, plus relationship and content-type wiring.

Usage:
    python docx_tool.py annotate workspace/ 0 "Comment text with &amp; entities"
    python docx_tool.py annotate workspace/ 1 "Reply" --reply-to 0
    python docx_tool.py annotate workspace/ 0 "Note" --author "Custom Name"

After running, manually place markers in document.xml:
    <w:commentRangeStart w:id="0"/>
    ... target content ...
    <w:commentRangeEnd w:id="0"/>
    <w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
"""

import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import defusedxml.minidom

FIXTURE_DIR = Path(__file__).parent / ".." / "fixtures"

WML_NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
    "w16cid": "http://schemas.microsoft.com/office/word/2016/wordml/cid",
    "w16cex": "http://schemas.microsoft.com/office/word/2018/wordml/cex",
}

ANNOTATION_FRAGMENT = """\
<w:comment w:id="{cid}" w:author="{author}" w:date="{timestamp}" w:initials="{initials}">
  <w:p w14:paraId="{paragraph_ref}" w14:textId="77777777">
    <w:r>
      <w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>
      <w:annotationRef/>
    </w:r>
    <w:r>
      <w:rPr>
        <w:color w:val="000000"/>
        <w:sz w:val="20"/>
        <w:szCs w:val="20"/>
      </w:rPr>
      <w:t>{body}</w:t>
    </w:r>
  </w:p>
</w:comment>"""

STANDALONE_MARKER_HINT = """
Place in document.xml (markers are siblings of w:r, never nested inside):
  <w:commentRangeStart w:id="{cid}"/>
  <w:r>...</w:r>
  <w:commentRangeEnd w:id="{cid}"/>
  <w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="{cid}"/></w:r>"""

THREADED_MARKER_HINT = """
Nest reply markers inside parent {pid}'s range (markers are siblings of w:r):
  <w:commentRangeStart w:id="{pid}"/><w:commentRangeStart w:id="{cid}"/>
  <w:r>...</w:r>
  <w:commentRangeEnd w:id="{cid}"/><w:commentRangeEnd w:id="{pid}"/>
  <w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="{pid}"/></w:r>
  <w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="{cid}"/></w:r>"""


def _random_hex_token() -> str:
    return f"{random.randint(0, 0x7FFFFFFE):08X}"


_CURLY_QUOTE_MAP = {
    "\u201c": "&#x201C;",
    "\u201d": "&#x201D;",
    "\u2018": "&#x2018;",
    "\u2019": "&#x2019;",
}


def _sanitize_quotes(text: str) -> str:
    for ch, ent in _CURLY_QUOTE_MAP.items():
        text = text.replace(ch, ent)
    return text


def _inject_fragment(xml_path: Path, container_tag: str, fragment: str) -> None:
    dom = defusedxml.minidom.parseString(xml_path.read_text(encoding="utf-8"))
    container = dom.getElementsByTagName(container_tag)[0]
    ns_decls = " ".join(f'xmlns:{k}="{v}"' for k, v in WML_NAMESPACES.items())
    wrapped = defusedxml.minidom.parseString(f"<_ns {ns_decls}>{fragment}</_ns>")
    for child in wrapped.documentElement.childNodes:
        if child.nodeType == child.ELEMENT_NODE:
            container.appendChild(dom.importNode(child, True))
    serialized = _sanitize_quotes(dom.toxml(encoding="UTF-8").decode("utf-8"))
    xml_path.write_text(serialized, encoding="utf-8")


def _locate_paragraph_ref(comments_path: Path, target_id: int) -> str | None:
    dom = defusedxml.minidom.parseString(comments_path.read_text(encoding="utf-8"))
    for comment_node in dom.getElementsByTagName("w:comment"):
        if comment_node.getAttribute("w:id") == str(target_id):
            for para in comment_node.getElementsByTagName("w:p"):
                ref = para.getAttribute("w14:paraId")
                if ref:
                    return ref
    return None


def _next_relationship_slot(rels_path: Path) -> int:
    dom = defusedxml.minidom.parseString(rels_path.read_text(encoding="utf-8"))
    highest = 0
    for rel in dom.getElementsByTagName("Relationship"):
        rid = rel.getAttribute("Id")
        if rid and rid.startswith("rId"):
            try:
                highest = max(highest, int(rid[3:]))
            except ValueError:
                pass
    return highest + 1


def _relationship_exists(rels_path: Path, target: str) -> bool:
    dom = defusedxml.minidom.parseString(rels_path.read_text(encoding="utf-8"))
    return any(
        r.getAttribute("Target") == target
        for r in dom.getElementsByTagName("Relationship")
    )


def _content_type_exists(ct_path: Path, part: str) -> bool:
    dom = defusedxml.minidom.parseString(ct_path.read_text(encoding="utf-8"))
    return any(
        o.getAttribute("PartName") == part
        for o in dom.getElementsByTagName("Override")
    )


def _wire_relationships(workspace: Path) -> None:
    rels_file = workspace / "word" / "_rels" / "document.xml.rels"
    if not rels_file.exists():
        return

    if _relationship_exists(rels_file, "comments.xml"):
        return

    dom = defusedxml.minidom.parseString(rels_file.read_text(encoding="utf-8"))
    root_elem = dom.documentElement
    slot = _next_relationship_slot(rels_file)

    bindings = [
        ("http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments",
         "comments.xml"),
        ("http://schemas.microsoft.com/office/2011/relationships/commentsExtended",
         "commentsExtended.xml"),
        ("http://schemas.microsoft.com/office/2016/09/relationships/commentsIds",
         "commentsIds.xml"),
        ("http://schemas.microsoft.com/office/2018/08/relationships/commentsExtensible",
         "commentsExtensible.xml"),
    ]

    for rel_type, target in bindings:
        node = dom.createElement("Relationship")
        node.setAttribute("Id", f"rId{slot}")
        node.setAttribute("Type", rel_type)
        node.setAttribute("Target", target)
        root_elem.appendChild(node)
        slot += 1

    rels_file.write_bytes(dom.toxml(encoding="UTF-8"))


def _wire_content_types(workspace: Path) -> None:
    ct_file = workspace / "[Content_Types].xml"
    if not ct_file.exists():
        return

    if _content_type_exists(ct_file, "/word/comments.xml"):
        return

    dom = defusedxml.minidom.parseString(ct_file.read_text(encoding="utf-8"))
    root_elem = dom.documentElement

    declarations = [
        ("/word/comments.xml",
         "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"),
        ("/word/commentsExtended.xml",
         "application/vnd.openxmlformats-officedocument.wordprocessingml.commentsExtended+xml"),
        ("/word/commentsIds.xml",
         "application/vnd.openxmlformats-officedocument.wordprocessingml.commentsIds+xml"),
        ("/word/commentsExtensible.xml",
         "application/vnd.openxmlformats-officedocument.wordprocessingml.commentsExtensible+xml"),
    ]

    for part_name, ct in declarations:
        node = dom.createElement("Override")
        node.setAttribute("PartName", part_name)
        node.setAttribute("ContentType", ct)
        root_elem.appendChild(node)

    ct_file.write_bytes(dom.toxml(encoding="UTF-8"))


def insert_annotation(
    workspace_path: str,
    annotation_id: int,
    body_text: str,
    author: str = "Verdent",
    initials: str = "V",
    parent_id: int | None = None,
) -> tuple[str, str]:
    word_dir = Path(workspace_path) / "word"
    if not word_dir.exists():
        return "", f"Error: {word_dir} not found"

    para_ref = _random_hex_token()
    durable_ref = _random_hex_token()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    comments_file = word_dir / "comments.xml"
    first_annotation = not comments_file.exists()
    if first_annotation:
        shutil.copy(FIXTURE_DIR / "comments.xml", comments_file)
        _wire_relationships(Path(workspace_path))
        _wire_content_types(Path(workspace_path))

    _inject_fragment(
        comments_file,
        "w:comments",
        ANNOTATION_FRAGMENT.format(
            cid=annotation_id,
            author=author,
            timestamp=now,
            initials=initials,
            paragraph_ref=para_ref,
            body=body_text,
        ),
    )

    ext_file = word_dir / "commentsExtended.xml"
    if not ext_file.exists():
        shutil.copy(FIXTURE_DIR / "commentsExtended.xml", ext_file)

    if parent_id is not None:
        parent_para = _locate_paragraph_ref(comments_file, parent_id)
        if not parent_para:
            return "", f"Error: Parent annotation {parent_id} not found"
        _inject_fragment(
            ext_file,
            "w15:commentsEx",
            f'<w15:commentEx w15:paraId="{para_ref}" w15:paraIdParent="{parent_para}" w15:done="0"/>',
        )
    else:
        _inject_fragment(
            ext_file,
            "w15:commentsEx",
            f'<w15:commentEx w15:paraId="{para_ref}" w15:done="0"/>',
        )

    ids_file = word_dir / "commentsIds.xml"
    if not ids_file.exists():
        shutil.copy(FIXTURE_DIR / "commentsIds.xml", ids_file)
    _inject_fragment(
        ids_file,
        "w16cid:commentsIds",
        f'<w16cid:commentId w16cid:paraId="{para_ref}" w16cid:durableId="{durable_ref}"/>',
    )

    extensible_file = word_dir / "commentsExtensible.xml"
    if not extensible_file.exists():
        shutil.copy(FIXTURE_DIR / "commentsExtensible.xml", extensible_file)
    _inject_fragment(
        extensible_file,
        "w16cex:commentsExtensible",
        f'<w16cex:commentExtensible w16cex:durableId="{durable_ref}" w16cex:dateUtc="{now}"/>',
    )

    kind = "reply" if parent_id is not None else "annotation"
    return para_ref, f"Added {kind} {annotation_id} (para_ref={para_ref})"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Insert annotations into unpacked DOCX")
    p.add_argument("workspace", help="Unpacked DOCX directory")
    p.add_argument("id", type=int, help="Annotation ID")
    p.add_argument("text", help="Annotation body text")
    p.add_argument("--author", default="Verdent")
    p.add_argument("--initials", default="V")
    p.add_argument("--reply-to", type=int, help="Parent annotation ID")
    args = p.parse_args()

    ref, msg = insert_annotation(
        args.workspace, args.id, args.text,
        author=args.author, initials=args.initials, parent_id=args.reply_to,
    )
    print(msg)
    if msg.startswith("Error"):
        sys.exit(1)

    cid = args.id
    if args.reply_to is not None:
        print(THREADED_MARKER_HINT.format(pid=args.reply_to, cid=cid))
    else:
        print(STANDALONE_MARKER_HINT.format(cid=cid))
