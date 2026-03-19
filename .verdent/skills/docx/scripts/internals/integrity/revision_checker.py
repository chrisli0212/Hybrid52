"""Validate that all document modifications are properly tracked.

Compares the document content after stripping one author's tracked changes
against the original to detect untracked edits. If text differs, the edit
was made outside the revision-tracking system.
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path


class RevisionIntegrityChecker:

    def __init__(self, unpacked_dir, original_archive, verbose=False, author="Verdent"):
        self.workspace = Path(unpacked_dir)
        self.original = Path(original_archive)
        self.verbose = verbose
        self.author = author
        self.wml = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def auto_repair(self) -> int:
        return 0

    def run_all(self):
        doc_xml = self.workspace / "word" / "document.xml"
        if not doc_xml.exists():
            print(f"FAILED - {doc_xml} not found")
            return False

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(doc_xml)
            root = tree.getroot()

            ns_author = f"{{{self.wml}}}author"
            has_changes = any(
                elem.get(ns_author) == self.author
                for elem in list(root.findall(f".//{{{self.wml}}}del", {"w": self.wml}))
                + list(root.findall(f".//{{{self.wml}}}ins", {"w": self.wml}))
            )

            if not has_changes:
                if self.verbose:
                    print(f"PASSED - No tracked changes by {self.author}")
                return True
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as td:
            try:
                with zipfile.ZipFile(self.original, "r") as zf:
                    zf.extractall(td)
            except Exception as e:
                print(f"FAILED - Cannot unpack original: {e}")
                return False

            orig_doc = Path(td) / "word" / "document.xml"
            if not orig_doc.exists():
                print(f"FAILED - Original document.xml not found in {self.original}")
                return False

            try:
                import xml.etree.ElementTree as ET
                mod_root = ET.parse(doc_xml).getroot()
                orig_root = ET.parse(orig_doc).getroot()
            except Exception as e:
                print(f"FAILED - XML parse error: {e}")
                return False

            self._strip_author_changes(orig_root)
            self._strip_author_changes(mod_root)

            text_mod = self._collect_text(mod_root)
            text_orig = self._collect_text(orig_root)

            if text_mod != text_orig:
                self._report_discrepancy(text_orig, text_mod)
                return False

            if self.verbose:
                print(f"PASSED - All edits by {self.author} properly tracked")
            return True

    def _strip_author_changes(self, root):
        ins_tag = f"{{{self.wml}}}ins"
        del_tag = f"{{{self.wml}}}del"
        author_key = f"{{{self.wml}}}author"

        for parent in root.iter():
            removals = [
                ch for ch in parent
                if ch.tag == ins_tag and ch.get(author_key) == self.author
            ]
            for elem in removals:
                parent.remove(elem)

        deltext_tag = f"{{{self.wml}}}delText"
        t_tag = f"{{{self.wml}}}t"

        for parent in root.iter():
            restorations = [
                (ch, list(parent).index(ch))
                for ch in parent
                if ch.tag == del_tag and ch.get(author_key) == self.author
            ]
            for del_elem, pos in reversed(restorations):
                for inner in del_elem.iter():
                    if inner.tag == deltext_tag:
                        inner.tag = t_tag
                for child in reversed(list(del_elem)):
                    parent.insert(pos, child)
                parent.remove(del_elem)

    def _collect_text(self, root):
        p_tag = f"{{{self.wml}}}p"
        t_tag = f"{{{self.wml}}}t"
        paragraphs = []
        for p in root.findall(f".//{p_tag}"):
            parts = [t.text for t in p.findall(f".//{t_tag}") if t.text]
            text = "".join(parts)
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)

    def _report_discrepancy(self, expected, actual):
        lines = [
            f"FAILED - Content mismatch after removing {self.author}'s tracked changes",
            "",
            "Possible causes:",
            "  1. Text modified inside another author's <w:ins> or <w:del>",
            "  2. Edits made without tracked-change wrappers",
            "  3. Incorrect nesting of <w:del> within <w:ins> for rejecting insertions",
            "",
            "Correction patterns:",
            "  - Reject another's insertion: nest <w:del> inside their <w:ins>",
            "  - Restore another's deletion: add <w:ins> after their <w:del>",
            "",
        ]

        diff = self._word_diff(expected, actual)
        if diff:
            lines.extend(["Differences:", "============", diff])
        else:
            lines.append("(git not available for diff)")

        print("\n".join(lines))

    def _word_diff(self, text_a, text_b):
        try:
            with tempfile.TemporaryDirectory() as td:
                fa = Path(td) / "a.txt"
                fb = Path(td) / "b.txt"
                fa.write_text(text_a, encoding="utf-8")
                fb.write_text(text_b, encoding="utf-8")

                for regex_opt in ("--word-diff-regex=.", None):
                    cmd = [
                        "git", "diff", "--word-diff=plain", "-U0",
                        "--no-index", str(fa), str(fb),
                    ]
                    if regex_opt:
                        cmd.insert(3, regex_opt)

                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.stdout.strip():
                        content = []
                        active = False
                        for line in proc.stdout.split("\n"):
                            if line.startswith("@@"):
                                active = True
                                continue
                            if active and line.strip():
                                content.append(line)
                        if content:
                            return "\n".join(content)

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return None
