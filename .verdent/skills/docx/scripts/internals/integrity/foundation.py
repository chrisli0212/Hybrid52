"""Foundation layer for Office document structural validation.

Provides reusable checks applicable across document types:
XML well-formedness, namespace consistency, ID uniqueness,
relationship integrity, content-type declarations, and XSD conformance.
"""

import re
from pathlib import Path

import defusedxml.minidom
import lxml.etree


class StructuralFoundation:

    SUPPRESSED_PATTERNS = [
        "hyphenationZone",
        "purl.org/dc/terms",
    ]

    ID_SCOPE_MAP = {
        "comment": ("id", "local"),
        "commentrangestart": ("id", "local"),
        "commentrangeend": ("id", "local"),
        "bookmarkstart": ("id", "local"),
        "bookmarkend": ("id", "local"),
        "sldid": ("id", "local"),
        "sldmasterid": ("id", "cross-file"),
        "sldlayoutid": ("id", "cross-file"),
        "cm": ("authorid", "local"),
        "sheet": ("sheetid", "local"),
        "definedname": ("id", "local"),
        "cxnsp": ("id", "local"),
        "sp": ("id", "local"),
        "pic": ("id", "local"),
        "grpsp": ("id", "local"),
    }

    SKIP_ID_PARENTS = {"sectionlst"}

    RELATIONSHIP_TYPE_HINTS = {}

    SCHEMA_ROUTES = {
        "word": "ISO-IEC29500-4_2016/wml.xsd",
        "ppt": "ISO-IEC29500-4_2016/pml.xsd",
        "xl": "ISO-IEC29500-4_2016/sml.xsd",
        "[Content_Types].xml": "ecma/fouth-edition/opc-contentTypes.xsd",
        "app.xml": "ISO-IEC29500-4_2016/shared-documentPropertiesExtended.xsd",
        "core.xml": "ecma/fouth-edition/opc-coreProperties.xsd",
        "custom.xml": "ISO-IEC29500-4_2016/shared-documentPropertiesCustom.xsd",
        ".rels": "ecma/fouth-edition/opc-relationships.xsd",
        "people.xml": "microsoft/wml-2012.xsd",
        "commentsIds.xml": "microsoft/wml-cid-2016.xsd",
        "commentsExtensible.xml": "microsoft/wml-cex-2018.xsd",
        "commentsExtended.xml": "microsoft/wml-2012.xsd",
        "chart": "ISO-IEC29500-4_2016/dml-chart.xsd",
        "theme": "ISO-IEC29500-4_2016/dml-main.xsd",
        "drawing": "ISO-IEC29500-4_2016/dml-main.xsd",
    }

    MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    XML_NS = "http://www.w3.org/XML/1998/namespace"
    PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

    CONTENT_ROOTS = {"word", "ppt", "xl"}

    OOXML_NS_SET = {
        "http://schemas.openxmlformats.org/officeDocument/2006/math",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "http://schemas.openxmlformats.org/schemaLibrary/2006/main",
        "http://schemas.openxmlformats.org/drawingml/2006/main",
        "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "http://schemas.openxmlformats.org/drawingml/2006/chartDrawing",
        "http://schemas.openxmlformats.org/drawingml/2006/diagram",
        "http://schemas.openxmlformats.org/drawingml/2006/picture",
        "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
        "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "http://schemas.openxmlformats.org/presentationml/2006/main",
        "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "http://schemas.openxmlformats.org/officeDocument/2006/sharedTypes",
        "http://www.w3.org/XML/1998/namespace",
    }

    def __init__(self, unpacked_dir, reference_file=None, verbose=False):
        self.root_dir = Path(unpacked_dir).resolve()
        self.reference = Path(reference_file) if reference_file else None
        self.verbose = verbose
        self.schemas_root = Path(__file__).parent.parent.parent / "schemas"

        self.markup_files = [
            f for ext in ("*.xml", "*.rels")
            for f in self.root_dir.rglob(ext)
        ]

        if not self.markup_files:
            print(f"Warning: No XML files found in {self.root_dir}")

    def run_all(self):
        raise NotImplementedError

    def auto_repair(self) -> int:
        return self._fix_whitespace_attrs()

    def _fix_whitespace_attrs(self) -> int:
        fixes = 0
        for xml_file in self.markup_files:
            try:
                raw = xml_file.read_text(encoding="utf-8")
                dom = defusedxml.minidom.parseString(raw)
                changed = False

                for elem in dom.getElementsByTagName("*"):
                    if elem.tagName.endswith(":t") and elem.firstChild:
                        val = elem.firstChild.nodeValue
                        if val and (val[0] in " \t" or val[-1] in " \t"):
                            if elem.getAttribute("xml:space") != "preserve":
                                elem.setAttribute("xml:space", "preserve")
                                fixes += 1
                                changed = True

                if changed:
                    xml_file.write_bytes(dom.toxml(encoding="UTF-8"))
            except Exception:
                pass
        return fixes

    def check_wellformedness(self):
        issues = []
        for f in self.markup_files:
            try:
                lxml.etree.parse(str(f))
            except lxml.etree.XMLSyntaxError as e:
                issues.append(f"  {f.relative_to(self.root_dir)}: Line {e.lineno}: {e.msg}")
            except Exception as e:
                issues.append(f"  {f.relative_to(self.root_dir)}: {e}")

        if issues:
            print(f"FAILED - {len(issues)} XML syntax error(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - All XML well-formed")
        return True

    def check_namespace_declarations(self):
        issues = []
        for f in self.markup_files:
            try:
                root = lxml.etree.parse(str(f)).getroot()
                declared = set(root.nsmap.keys()) - {None}
                for attr_val in [v for k, v in root.attrib.items() if k.endswith("Ignorable")]:
                    undeclared = set(attr_val.split()) - declared
                    for ns in undeclared:
                        issues.append(f"  {f.relative_to(self.root_dir)}: '{ns}' in Ignorable but undeclared")
            except lxml.etree.XMLSyntaxError:
                continue

        if issues:
            print(f"FAILED - {len(issues)} namespace issue(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - Namespace declarations consistent")
        return True

    def check_id_uniqueness(self):
        issues = []
        cross_file_ids = {}

        for f in self.markup_files:
            try:
                root = lxml.etree.parse(str(f)).getroot()
                local_ids = {}

                mc_nodes = root.xpath(".//mc:AlternateContent", namespaces={"mc": self.MC_NS})
                for node in mc_nodes:
                    node.getparent().remove(node)

                for elem in root.iter():
                    tag = (elem.tag.split("}")[-1].lower() if "}" in elem.tag else elem.tag.lower())

                    if tag in self.ID_SCOPE_MAP:
                        if any(
                            anc.tag.split("}")[-1].lower() in self.SKIP_ID_PARENTS
                            for anc in elem.iterancestors()
                        ):
                            continue

                        attr_name, scope = self.ID_SCOPE_MAP[tag]
                        id_val = None
                        for a, v in elem.attrib.items():
                            local_a = a.split("}")[-1].lower() if "}" in a else a.lower()
                            if local_a == attr_name:
                                id_val = v
                                break

                        if id_val is None:
                            continue

                        if scope == "cross-file":
                            if id_val in cross_file_ids:
                                prev_f, prev_l, prev_t = cross_file_ids[id_val]
                                issues.append(
                                    f"  {f.relative_to(self.root_dir)}: Line {elem.sourceline}: "
                                    f"Cross-file ID '{id_val}' in <{tag}> duplicates "
                                    f"{prev_f} line {prev_l} <{prev_t}>"
                                )
                            else:
                                cross_file_ids[id_val] = (
                                    f.relative_to(self.root_dir), elem.sourceline, tag
                                )
                        else:
                            key = (tag, attr_name)
                            if key not in local_ids:
                                local_ids[key] = {}
                            if id_val in local_ids[key]:
                                issues.append(
                                    f"  {f.relative_to(self.root_dir)}: Line {elem.sourceline}: "
                                    f"Duplicate {attr_name}='{id_val}' in <{tag}> "
                                    f"(first at line {local_ids[key][id_val]})"
                                )
                            else:
                                local_ids[key][id_val] = elem.sourceline

            except Exception as e:
                issues.append(f"  {f.relative_to(self.root_dir)}: {e}")

        if issues:
            print(f"FAILED - {len(issues)} ID uniqueness violation(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - All IDs unique")
        return True

    def check_relationship_targets(self):
        issues = []
        rels_files = list(self.root_dir.rglob("*.rels"))

        if not rels_files:
            if self.verbose:
                print("PASSED - No .rels files")
            return True

        all_files = {
            p.resolve()
            for p in self.root_dir.rglob("*")
            if p.is_file() and p.name != "[Content_Types].xml" and not p.name.endswith(".rels")
        }

        referenced = set()

        for rf in rels_files:
            try:
                root = lxml.etree.parse(str(rf)).getroot()
                for rel in root.findall(f".//{{{self.PKG_REL_NS}}}Relationship"):
                    target = rel.get("Target")
                    if not target or target.startswith(("http", "mailto:")):
                        continue

                    if target.startswith("/"):
                        resolved = self.root_dir / target.lstrip("/")
                    elif rf.name == ".rels":
                        resolved = self.root_dir / target
                    else:
                        resolved = rf.parent.parent / target

                    try:
                        resolved = resolved.resolve()
                        if resolved.exists() and resolved.is_file():
                            referenced.add(resolved)
                        else:
                            issues.append(
                                f"  {rf.relative_to(self.root_dir)}: Line {rel.sourceline}: "
                                f"Broken reference -> {target}"
                            )
                    except (OSError, ValueError):
                        issues.append(
                            f"  {rf.relative_to(self.root_dir)}: Line {rel.sourceline}: "
                            f"Invalid path -> {target}"
                        )
            except Exception as e:
                issues.append(f"  Error parsing {rf.relative_to(self.root_dir)}: {e}")

        orphans = all_files - referenced
        for orphan in sorted(orphans):
            issues.append(f"  Unreferenced: {orphan.relative_to(self.root_dir)}")

        if issues:
            print(f"FAILED - {len(issues)} relationship issue(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - All relationships valid")
        return True

    def check_relationship_ids(self):
        issues = []

        for xml_file in self.markup_files:
            if xml_file.suffix == ".rels":
                continue

            rels_file = xml_file.parent / "_rels" / f"{xml_file.name}.rels"
            if not rels_file.exists():
                continue

            try:
                rels_root = lxml.etree.parse(str(rels_file)).getroot()
                known_rids = {}

                for rel in rels_root.findall(f"{{{self.PKG_REL_NS}}}Relationship"):
                    rid = rel.get("Id")
                    rtype = rel.get("Type", "")
                    if rid:
                        if rid in known_rids:
                            issues.append(
                                f"  {rels_file.relative_to(self.root_dir)}: "
                                f"Line {rel.sourceline}: Duplicate rId '{rid}'"
                            )
                        known_rids[rid] = rtype.split("/")[-1] if "/" in rtype else rtype

                xml_root = lxml.etree.parse(str(xml_file)).getroot()
                for elem in xml_root.iter():
                    for check_attr in ("id", "embed", "link"):
                        ref = elem.get(f"{{{self.DOC_REL_NS}}}{check_attr}")
                        if ref and ref not in known_rids:
                            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                            issues.append(
                                f"  {xml_file.relative_to(self.root_dir)}: "
                                f"Line {elem.sourceline}: <{tag}> references missing rId '{ref}'"
                            )

            except Exception as e:
                issues.append(f"  Error processing {xml_file.relative_to(self.root_dir)}: {e}")

        if issues:
            print(f"FAILED - {len(issues)} rId reference error(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - All rId references valid")
        return True

    def check_content_types(self):
        issues = []
        ct_file = self.root_dir / "[Content_Types].xml"

        if not ct_file.exists():
            print("FAILED - [Content_Types].xml missing")
            return False

        try:
            root = lxml.etree.parse(str(ct_file)).getroot()
            declared_parts = {
                o.get("PartName").lstrip("/")
                for o in root.findall(f"{{{self.CT_NS}}}Override")
                if o.get("PartName")
            }
            declared_exts = {
                d.get("Extension").lower()
                for d in root.findall(f"{{{self.CT_NS}}}Default")
                if d.get("Extension")
            }

            media_types = {
                "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff",
                "wmf": "image/x-wmf", "emf": "image/x-emf",
            }

            for f in self.root_dir.rglob("*"):
                if not f.is_file() or f.suffix.lower() in (".xml", ".rels"):
                    continue
                if f.name == "[Content_Types].xml":
                    continue
                if "_rels" in f.parts or "docProps" in f.parts:
                    continue

                ext = f.suffix.lstrip(".").lower()
                if ext and ext not in declared_exts and ext in media_types:
                    issues.append(
                        f"  {f.relative_to(self.root_dir)}: Extension '{ext}' not declared"
                    )

        except Exception as e:
            issues.append(f"  Error parsing [Content_Types].xml: {e}")

        if issues:
            print(f"FAILED - {len(issues)} content-type issue(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - Content types declared")
        return True

    def check_xsd_conformance(self):
        new_issues = []

        for xml_file in self.markup_files:
            valid, errors = self._validate_one_xsd(xml_file)

            if valid is None:
                continue
            if valid:
                continue

            rel = xml_file.relative_to(self.root_dir)
            new_issues.append(f"  {rel}: {len(errors)} new error(s)")
            for err in list(errors)[:3]:
                truncated = err[:250] + "..." if len(err) > 250 else err
                new_issues.append(f"    - {truncated}")

        if new_issues:
            print("\nFAILED - New XSD validation errors:")
            for i in new_issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - No new XSD errors")
        return True

    def _resolve_schema(self, xml_file):
        if xml_file.name in self.SCHEMA_ROUTES:
            return self.schemas_root / self.SCHEMA_ROUTES[xml_file.name]
        if xml_file.suffix == ".rels":
            return self.schemas_root / self.SCHEMA_ROUTES[".rels"]
        if "charts/" in str(xml_file) and xml_file.name.startswith("chart"):
            return self.schemas_root / self.SCHEMA_ROUTES["chart"]
        if "theme/" in str(xml_file) and xml_file.name.startswith("theme"):
            return self.schemas_root / self.SCHEMA_ROUTES["theme"]
        if xml_file.parent.name in self.CONTENT_ROOTS:
            return self.schemas_root / self.SCHEMA_ROUTES[xml_file.parent.name]
        return None

    def _strip_non_ooxml(self, doc):
        raw = lxml.etree.tostring(doc, encoding="unicode")
        copy = lxml.etree.fromstring(raw)

        for elem in copy.iter():
            to_drop = [
                a for a in elem.attrib
                if "{" in a and a.split("}")[0][1:] not in self.OOXML_NS_SET
            ]
            for a in to_drop:
                del elem.attrib[a]

        self._prune_foreign_elements(copy)
        return lxml.etree.ElementTree(copy)

    def _prune_foreign_elements(self, root):
        to_remove = []
        for elem in list(root):
            if not hasattr(elem, "tag") or callable(elem.tag):
                continue
            tag = str(elem.tag)
            if tag.startswith("{"):
                ns = tag.split("}")[0][1:]
                if ns not in self.OOXML_NS_SET:
                    to_remove.append(elem)
                    continue
            self._prune_foreign_elements(elem)
        for elem in to_remove:
            root.remove(elem)

    def _strip_template_expressions(self, doc):
        pattern = re.compile(r"\{\{[^}]*\}\}")
        raw = lxml.etree.tostring(doc, encoding="unicode")
        copy = lxml.etree.fromstring(raw)

        for elem in copy.iter():
            if not hasattr(elem, "tag") or callable(elem.tag):
                continue
            tag = str(elem.tag)
            if tag.endswith("}t") or tag == "t":
                continue
            if elem.text:
                elem.text = pattern.sub("", elem.text)
            if elem.tail:
                elem.tail = pattern.sub("", elem.tail)

        return lxml.etree.ElementTree(copy)

    def _validate_one_xsd(self, xml_file):
        schema_path = self._resolve_schema(xml_file)
        if not schema_path:
            return None, None

        try:
            xsd_doc = lxml.etree.parse(str(schema_path), base_url=str(schema_path))
            schema = lxml.etree.XMLSchema(xsd_doc)

            xml_doc = lxml.etree.parse(str(xml_file))
            xml_doc = self._strip_template_expressions(xml_doc)

            mc_attr = f"{{{self.MC_NS}}}Ignorable"
            if mc_attr in xml_doc.getroot().attrib:
                del xml_doc.getroot().attrib[mc_attr]

            rel = xml_file.relative_to(self.root_dir)
            if rel.parts and rel.parts[0] in self.CONTENT_ROOTS:
                xml_doc = self._strip_non_ooxml(xml_doc)

            if schema.validate(xml_doc):
                return True, set()

            current = {err.message for err in schema.error_log}
            baseline = self._baseline_errors(xml_file)
            fresh = current - baseline
            fresh = {
                e for e in fresh
                if not any(p in e for p in self.SUPPRESSED_PATTERNS)
            }

            return (not fresh), fresh

        except Exception as e:
            err_str = str(e)
            if any(p in err_str for p in self.SUPPRESSED_PATTERNS):
                return True, set()
            return False, {err_str}

    def _baseline_errors(self, xml_file):
        if not self.reference:
            return set()

        import tempfile
        import zipfile

        rel = xml_file.relative_to(self.root_dir)
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(self.reference, "r") as zf:
                zf.extractall(td)
            original = Path(td) / rel
            if not original.exists():
                return set()
            _, errs = self._validate_one_xsd.__wrapped__(self, original) if hasattr(self._validate_one_xsd, '__wrapped__') else (None, set())
            # Simplified: just re-validate the original file
            schema_path = self._resolve_schema(original)
            if not schema_path:
                return set()
            try:
                xsd_doc = lxml.etree.parse(str(schema_path), base_url=str(schema_path))
                schema = lxml.etree.XMLSchema(xsd_doc)
                xml_doc = lxml.etree.parse(str(original))
                schema.validate(xml_doc)
                return {err.message for err in schema.error_log}
            except Exception:
                return set()
