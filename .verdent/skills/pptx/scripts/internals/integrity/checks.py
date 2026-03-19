"""Structural integrity checks for unpacked PPTX workspaces.

Validates well-formed XML, namespace declarations, element ID uniqueness,
relationship references, content types, and PPTX-specific constraints.
"""
from __future__ import annotations

import re
from pathlib import Path

import defusedxml.minidom
import lxml.etree

_PKG_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_RELS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_PML_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


class DeckIntegrityChecker:

    _ELEMENT_ID_RULES = {
        "sldid": ("id", "file"),
        "sldmasterid": ("id", "global"),
        "sldlayoutid": ("id", "global"),
        "sp": ("id", "file"),
        "pic": ("id", "file"),
        "cxnsp": ("id", "file"),
        "grpsp": ("id", "file"),
    }

    _SKIP_ANCESTORS = {"sectionlst"}

    def __init__(self, workspace: Path, reference: Path | None = None):
        self.root = workspace.resolve()
        self.reference = reference
        self.xml_files = [
            f for pat in ("*.xml", "*.rels")
            for f in self.root.rglob(pat)
        ]

    def auto_repair(self) -> int:
        count = 0
        for xf in self.xml_files:
            try:
                raw = xf.read_text(encoding="utf-8")
                dom = defusedxml.minidom.parseString(raw)
                touched = False
                for el in dom.getElementsByTagName("*"):
                    if el.tagName.endswith(":t") and el.firstChild:
                        val = el.firstChild.nodeValue
                        if val and (val[0] in " \t" or val[-1] in " \t"):
                            if el.getAttribute("xml:space") != "preserve":
                                el.setAttribute("xml:space", "preserve")
                                count += 1
                                touched = True
                if touched:
                    xf.write_bytes(dom.toxml(encoding="UTF-8"))
            except Exception:
                pass
        return count

    def run_all(self) -> bool:
        if not self._check_xml_syntax():
            return False
        return all([
            self._check_ns_declarations(),
            self._check_element_ids(),
            self._check_rels_targets(),
            self._check_content_types(),
            self._check_layout_bindings(),
            self._check_single_layout_per_slide(),
            self._check_notes_not_shared(),
        ])

    def _check_xml_syntax(self) -> bool:
        errors = []
        for xf in self.xml_files:
            try:
                lxml.etree.parse(str(xf))
            except lxml.etree.XMLSyntaxError as e:
                errors.append(f"  {xf.relative_to(self.root)}: L{e.lineno}: {e.msg}")
        if errors:
            print(f"XML syntax errors ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_ns_declarations(self) -> bool:
        errors = []
        for xf in self.xml_files:
            try:
                root = lxml.etree.parse(str(xf)).getroot()
                declared = set(root.nsmap.keys()) - {None}
                for v in [v for k, v in root.attrib.items() if k.endswith("Ignorable")]:
                    for pfx in set(v.split()) - declared:
                        errors.append(
                            f"  {xf.relative_to(self.root)}: "
                            f"Undeclared prefix '{pfx}' in mc:Ignorable"
                        )
            except lxml.etree.XMLSyntaxError:
                continue
        if errors:
            print(f"Namespace issues ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_element_ids(self) -> bool:
        errors = []
        global_ids: dict[str, tuple] = {}

        for xf in self.xml_files:
            try:
                root = lxml.etree.parse(str(xf)).getroot()
                local_ids: dict[tuple, dict] = {}

                for mc in root.xpath(".//mc:AlternateContent", namespaces={"mc": _MC_NS}):
                    mc.getparent().remove(mc)

                for el in root.iter():
                    tag = el.tag.split("}")[-1].lower() if "}" in el.tag else el.tag.lower()
                    if tag not in self._ELEMENT_ID_RULES:
                        continue
                    if any(
                        a.tag.split("}")[-1].lower() in self._SKIP_ANCESTORS
                        for a in el.iterancestors()
                    ):
                        continue

                    attr_name, scope = self._ELEMENT_ID_RULES[tag]
                    val = None
                    for a, v in el.attrib.items():
                        if (a.split("}")[-1].lower() if "}" in a else a.lower()) == attr_name:
                            val = v
                            break
                    if val is None:
                        continue

                    rp = xf.relative_to(self.root)
                    if scope == "global":
                        if val in global_ids:
                            pf, pl, pt = global_ids[val]
                            errors.append(
                                f"  {rp}: L{el.sourceline}: Global ID '{val}' in <{tag}> "
                                f"duplicates {pf} L{pl} <{pt}>"
                            )
                        else:
                            global_ids[val] = (rp, el.sourceline, tag)
                    else:
                        key = (tag, attr_name)
                        if key not in local_ids:
                            local_ids[key] = {}
                        if val in local_ids[key]:
                            errors.append(
                                f"  {rp}: L{el.sourceline}: Dup {attr_name}='{val}' in <{tag}>"
                            )
                        else:
                            local_ids[key][val] = el.sourceline
            except Exception as e:
                errors.append(f"  {xf.relative_to(self.root)}: {e}")

        if errors:
            print(f"ID uniqueness violations ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_rels_targets(self) -> bool:
        errors = []
        for rf in self.root.rglob("*.rels"):
            try:
                root = lxml.etree.parse(str(rf)).getroot()
                for rel in root.findall(f".//{{{_PKG_RELS}}}Relationship"):
                    target = rel.get("Target")
                    if not target or target.startswith(("http", "mailto:")):
                        continue
                    if target.startswith("/"):
                        tp = self.root / target.lstrip("/")
                    elif rf.name == ".rels":
                        tp = self.root / target
                    else:
                        tp = rf.parent.parent / target
                    tp = tp.resolve()
                    if not tp.exists():
                        errors.append(
                            f"  {rf.relative_to(self.root)}: Broken -> {target}"
                        )
            except Exception as e:
                errors.append(f"  {rf.relative_to(self.root)}: {e}")

        if errors:
            print(f"Broken references ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_content_types(self) -> bool:
        ct_file = self.root / "[Content_Types].xml"
        if not ct_file.exists():
            print("[Content_Types].xml missing")
            return False

        errors = []
        try:
            root = lxml.etree.parse(str(ct_file)).getroot()
            declared = {
                ov.get("PartName", "").lstrip("/")
                for ov in root.findall(f".//{{{_CT_NS}}}Override")
            }
            known_exts = {
                d.get("Extension", "").lower()
                for d in root.findall(f".//{{{_CT_NS}}}Default")
            }

            important_roots = {"sld", "sldLayout", "sldMaster", "presentation", "theme"}
            for xf in self.xml_files:
                if xf.suffix == ".rels" or "[Content_Types]" in xf.name:
                    continue
                rp = str(xf.relative_to(self.root)).replace("\\", "/")
                if "docProps/" in rp or "_rels/" in rp:
                    continue
                try:
                    rt = lxml.etree.parse(str(xf)).getroot().tag
                    rn = rt.split("}")[-1] if "}" in rt else rt
                    if rn in important_roots and rp not in declared:
                        errors.append(f"  {rp}: not in [Content_Types].xml")
                except Exception:
                    continue

        except Exception as e:
            errors.append(f"  Parse error: {e}")

        if errors:
            print(f"Content type issues ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_layout_bindings(self) -> bool:
        errors = []
        for sm in self.root.glob("ppt/slideMasters/*.xml"):
            try:
                root = lxml.etree.parse(str(sm)).getroot()
                rf = sm.parent / "_rels" / f"{sm.name}.rels"
                if not rf.exists():
                    errors.append(f"  {sm.relative_to(self.root)}: No .rels file")
                    continue

                rroot = lxml.etree.parse(str(rf)).getroot()
                valid = {
                    rel.get("Id")
                    for rel in rroot.findall(f".//{{{_PKG_RELS}}}Relationship")
                    if "slideLayout" in rel.get("Type", "")
                }

                for lid in root.findall(f".//{{{_PML_NS}}}sldLayoutId"):
                    rid = lid.get(f"{{{_OFFICE_RELS}}}id")
                    if rid and rid not in valid:
                        errors.append(
                            f"  {sm.relative_to(self.root)}: L{lid.sourceline}: "
                            f"sldLayoutId r:id='{rid}' has no matching relationship"
                        )
            except Exception as e:
                errors.append(f"  {sm.relative_to(self.root)}: {e}")

        if errors:
            print(f"Layout binding errors ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_single_layout_per_slide(self) -> bool:
        errors = []
        for rf in self.root.glob("ppt/slides/_rels/*.xml.rels"):
            try:
                root = lxml.etree.parse(str(rf)).getroot()
                layouts = [
                    r for r in root.findall(f".//{{{_PKG_RELS}}}Relationship")
                    if "slideLayout" in r.get("Type", "")
                ]
                if len(layouts) > 1:
                    errors.append(
                        f"  {rf.relative_to(self.root)}: {len(layouts)} layout refs (expected 1)"
                    )
            except Exception as e:
                errors.append(f"  {rf.relative_to(self.root)}: {e}")

        if errors:
            print(f"Multiple-layout slides ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors

    def _check_notes_not_shared(self) -> bool:
        errors = []
        target_owners: dict[str, list[str]] = {}
        for rf in self.root.glob("ppt/slides/_rels/*.xml.rels"):
            try:
                root = lxml.etree.parse(str(rf)).getroot()
                for rel in root.findall(f".//{{{_PKG_RELS}}}Relationship"):
                    if "notesSlide" in rel.get("Type", ""):
                        t = rel.get("Target", "").replace("../", "")
                        owner = rf.stem.replace(".xml", "")
                        target_owners.setdefault(t, []).append(owner)
            except Exception as e:
                errors.append(f"  {rf.relative_to(self.root)}: {e}")

        for target, owners in target_owners.items():
            if len(owners) > 1:
                errors.append(f"  '{target}' shared by: {', '.join(owners)}")

        if errors:
            print(f"Shared notes slides ({len(errors)}):")
            for e in errors:
                print(e)
        return not errors
