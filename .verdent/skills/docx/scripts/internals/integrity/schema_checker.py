"""Word-specific document validation extending the structural foundation.

Checks whitespace preservation on text elements, correct use of deletion
and insertion markup, paragraph count comparison, ID range constraints,
and comment marker integrity.
"""

import random
import re
import tempfile
import zipfile

import defusedxml.minidom
import lxml.etree

from .foundation import StructuralFoundation


class WordDocumentChecker(StructuralFoundation):

    WML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
    W16CID_NS = "http://schemas.microsoft.com/office/word/2016/wordml/cid"

    def run_all(self):
        if not self.check_wellformedness():
            return False

        passed = True
        for check in (
            self.check_namespace_declarations,
            self.check_id_uniqueness,
            self.check_relationship_targets,
            self.check_content_types,
            self.check_xsd_conformance,
            self.check_whitespace_attrs,
            self.check_deletion_markup,
            self.check_insertion_markup,
            self.check_relationship_ids,
            self.check_id_value_ranges,
            self.check_comment_markers,
        ):
            if not check():
                passed = False

        self._report_paragraph_delta()
        return passed

    def check_whitespace_attrs(self):
        issues = []
        for f in self.markup_files:
            if f.name != "document.xml":
                continue
            try:
                root = lxml.etree.parse(str(f)).getroot()
                for elem in root.iter(f"{{{self.WML_NS}}}t"):
                    if elem.text:
                        txt = elem.text
                        if re.search(r"^[ \t\n\r]", txt) or re.search(r"[ \t\n\r]$", txt):
                            space_attr = f"{{{self.XML_NS}}}space"
                            if space_attr not in elem.attrib or elem.attrib[space_attr] != "preserve":
                                preview = repr(txt)[:50]
                                issues.append(
                                    f"  {f.relative_to(self.root_dir)}: "
                                    f"Line {elem.sourceline}: Missing xml:space='preserve' on: {preview}"
                                )
            except Exception as e:
                issues.append(f"  {f.relative_to(self.root_dir)}: {e}")

        if issues:
            print(f"FAILED - {len(issues)} whitespace preservation issue(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - Whitespace preservation correct")
        return True

    def check_deletion_markup(self):
        issues = []
        ns = {"w": self.WML_NS}

        for f in self.markup_files:
            if f.name != "document.xml":
                continue
            try:
                root = lxml.etree.parse(str(f)).getroot()
                for t_elem in root.xpath(".//w:del//w:t", namespaces=ns):
                    if t_elem.text:
                        preview = repr(t_elem.text)[:50]
                        issues.append(
                            f"  {f.relative_to(self.root_dir)}: "
                            f"Line {t_elem.sourceline}: <w:t> inside <w:del> (use <w:delText>): {preview}"
                        )
                for instr in root.xpath(".//w:del//w:instrText", namespaces=ns):
                    issues.append(
                        f"  {f.relative_to(self.root_dir)}: "
                        f"Line {instr.sourceline}: <w:instrText> inside <w:del> (use <w:delInstrText>)"
                    )
            except Exception as e:
                issues.append(f"  {f.relative_to(self.root_dir)}: {e}")

        if issues:
            print(f"FAILED - {len(issues)} deletion markup error(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - Deletion markup correct")
        return True

    def check_insertion_markup(self):
        issues = []
        ns = {"w": self.WML_NS}

        for f in self.markup_files:
            if f.name != "document.xml":
                continue
            try:
                root = lxml.etree.parse(str(f)).getroot()
                bad = root.xpath(".//w:ins//w:delText[not(ancestor::w:del)]", namespaces=ns)
                for elem in bad:
                    preview = repr(elem.text or "")[:50]
                    issues.append(
                        f"  {f.relative_to(self.root_dir)}: "
                        f"Line {elem.sourceline}: <w:delText> inside <w:ins>: {preview}"
                    )
            except Exception as e:
                issues.append(f"  {f.relative_to(self.root_dir)}: {e}")

        if issues:
            print(f"FAILED - {len(issues)} insertion markup error(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - Insertion markup correct")
        return True

    def check_id_value_ranges(self):
        issues = []
        para_id_key = f"{{{self.W14_NS}}}paraId"
        durable_key = f"{{{self.W16CID_NS}}}durableId"

        for f in self.markup_files:
            try:
                for elem in lxml.etree.parse(str(f)).iter():
                    if val := elem.get(para_id_key):
                        if int(val, 16) >= 0x80000000:
                            issues.append(f"  {f.name}:{elem.sourceline}: paraId={val} out of range")

                    if val := elem.get(durable_key):
                        if f.name == "numbering.xml":
                            try:
                                if int(val, 10) >= 0x7FFFFFFF:
                                    issues.append(f"  {f.name}:{elem.sourceline}: durableId={val} out of range")
                            except ValueError:
                                issues.append(f"  {f.name}:{elem.sourceline}: durableId must be decimal")
                        else:
                            if int(val, 16) >= 0x7FFFFFFF:
                                issues.append(f"  {f.name}:{elem.sourceline}: durableId={val} out of range")
            except Exception:
                pass

        if issues:
            print(f"FAILED - {len(issues)} ID range violation(s):")
            for i in issues:
                print(i)
        elif self.verbose:
            print("PASSED - All ID values within allowed ranges")
        return not issues

    def check_comment_markers(self):
        issues = []
        doc_xml = comments_xml = None
        for f in self.markup_files:
            if f.name == "document.xml" and "word" in str(f):
                doc_xml = f
            elif f.name == "comments.xml":
                comments_xml = f

        if not doc_xml:
            if self.verbose:
                print("PASSED - No document.xml (skipping comment check)")
            return True

        try:
            ns = {"w": self.WML_NS}
            root = lxml.etree.parse(str(doc_xml)).getroot()

            starts = {e.get(f"{{{self.WML_NS}}}id") for e in root.xpath(".//w:commentRangeStart", namespaces=ns)}
            ends = {e.get(f"{{{self.WML_NS}}}id") for e in root.xpath(".//w:commentRangeEnd", namespaces=ns)}
            refs = {e.get(f"{{{self.WML_NS}}}id") for e in root.xpath(".//w:commentReference", namespaces=ns)}

            for cid in sorted(ends - starts, key=lambda x: int(x) if x and x.isdigit() else 0):
                issues.append(f'  document.xml: commentRangeEnd id="{cid}" without matching Start')
            for cid in sorted(starts - ends, key=lambda x: int(x) if x and x.isdigit() else 0):
                issues.append(f'  document.xml: commentRangeStart id="{cid}" without matching End')

            if comments_xml and comments_xml.exists():
                croot = lxml.etree.parse(str(comments_xml)).getroot()
                known = {e.get(f"{{{self.WML_NS}}}id") for e in croot.xpath(".//w:comment", namespaces=ns)}
                phantom = (starts | ends | refs) - known
                for cid in sorted(phantom, key=lambda x: int(x) if x and x.isdigit() else 0):
                    if cid:
                        issues.append(f'  document.xml: marker id="{cid}" references missing comment')

        except Exception as e:
            issues.append(f"  Error: {e}")

        if issues:
            print(f"FAILED - {len(issues)} comment marker issue(s):")
            for i in issues:
                print(i)
            return False
        if self.verbose:
            print("PASSED - Comment markers consistent")
        return True

    def _count_paragraphs(self, root):
        return len(root.findall(f".//{{{self.WML_NS}}}p"))

    def _report_paragraph_delta(self):
        if not self.reference:
            return

        try:
            current_root = None
            for f in self.markup_files:
                if f.name == "document.xml":
                    current_root = lxml.etree.parse(str(f)).getroot()
                    break

            if current_root is None:
                return

            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(self.reference, "r") as zf:
                    zf.extractall(td)
                orig = lxml.etree.parse(f"{td}/word/document.xml").getroot()

            n_orig = self._count_paragraphs(orig)
            n_curr = self._count_paragraphs(current_root)
            delta = n_curr - n_orig
            sign = f"+{delta}" if delta > 0 else str(delta)
            print(f"\nParagraphs: {n_orig} -> {n_curr} ({sign})")
        except Exception:
            pass

    def auto_repair(self) -> int:
        fixes = super().auto_repair()
        fixes += self._fix_durable_ids()
        return fixes

    def _fix_durable_ids(self) -> int:
        fixes = 0
        for f in self.markup_files:
            try:
                raw = f.read_text(encoding="utf-8")
                dom = defusedxml.minidom.parseString(raw)
                changed = False

                for elem in dom.getElementsByTagName("*"):
                    if not elem.hasAttribute("w16cid:durableId"):
                        continue

                    val = elem.getAttribute("w16cid:durableId")
                    bad = False

                    if f.name == "numbering.xml":
                        try:
                            bad = int(val, 10) >= 0x7FFFFFFF
                        except ValueError:
                            bad = True
                    else:
                        try:
                            bad = int(val, 16) >= 0x7FFFFFFF
                        except ValueError:
                            bad = True

                    if bad:
                        replacement = random.randint(1, 0x7FFFFFFE)
                        new_val = str(replacement) if f.name == "numbering.xml" else f"{replacement:08X}"
                        elem.setAttribute("w16cid:durableId", new_val)
                        print(f"  Fixed: {f.name}: durableId {val} -> {new_val}")
                        fixes += 1
                        changed = True

                if changed:
                    f.write_bytes(dom.toxml(encoding="UTF-8"))
            except Exception:
                pass

        return fixes
