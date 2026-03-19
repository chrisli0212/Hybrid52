---
name: docx
description: 'Word document toolkit for .docx files. Activate when the task involves reading, creating, editing, analyzing, or converting Word documents. Triggers on mentions of .docx, Word files, reports, memos, letters, proposals, contracts, or any request to produce formatted documents with headings, tables of contents, page numbers, or professional layouts. Also covers tracked changes, comments, find-and-replace in Word XML, and image insertion. Do not use for PDFs, spreadsheets, or plain text files.'
metadata:
  version: '1.1.0'
---

# Word Document Toolkit

Determine the goal, then follow the matching path.

**Reading or analyzing content?** Jump to [Content Extraction](#content-extraction).
**Creating a new document from scratch?** Load [references/creation-guide.md](references/creation-guide.md).
**Editing an existing document?** Load [references/xml-editing-guide.md](references/xml-editing-guide.md).

---

## Content Extraction

### Text Extraction

```bash
pandoc --track-changes=all document.docx -o output.md

python scripts/docx_tool.py extract document.docx unpacked/
```

### Visual Preview

Render pages as images for layout inspection:

```bash
python scripts/docx_tool.py render document.docx
python scripts/docx_tool.py render document.docx --range 1-3 --dpi 200 --dest previews/
```

Converts to PDF via LibreOffice, then rasterizes with Poppler. The runtime adapts to sandboxed environments automatically.

### Legacy Format Conversion

Convert `.doc` files before processing:

```bash
python scripts/docx_tool.py convert document.doc --to docx
```

### Finalizing Tracked Changes

Produce a clean copy with all revisions accepted:

```bash
python scripts/docx_tool.py accept-changes input.docx output.docx
```

---

## Verification Protocol (Mandatory)

After generating or modifying any document, always validate before delivering.

### Step 1: Structural Validation

```bash
python scripts/docx_tool.py verify output.docx
python scripts/docx_tool.py verify unpacked/ --original source.docx
```

Checks XML well-formedness, namespace declarations, unique IDs, relationship integrity, content type declarations, XSD schema conformance, whitespace preservation, tracked change correctness, and comment marker pairing.

### Step 2: Content Verification

```bash
python scripts/docx_tool.py extract output.docx --format text | grep -iE "placeholder|lorem|xxxx|TODO"
```

Confirm no leftover template content, missing sections, or incorrect ordering.

### Step 3: Visual QA via Subagent

```bash
python scripts/docx_tool.py render output.docx
```

Dispatch a subagent to inspect rendered pages:

```
These document page images may contain formatting issues. Identify them.

Checklist:
- Missing or incorrectly ordered headings
- Table layout problems (misaligned columns, missing borders)
- Image sizing or positioning errors
- Inconsistent fonts or spacing
- Page break placement issues
- Header/footer rendering problems

Report each issue with page number and severity.
```

### Step 4: Fix-and-Verify Loop

1. Generate/Edit -> Validate -> Render -> Inspect
2. Catalog all defects
3. Fix each issue
4. Re-validate and re-render affected pages
5. Repeat until a full pass finds zero new issues

Do not consider the task complete without at least one full verification cycle.

---

## Bundled Scripts (unified CLI)

All operations go through `scripts/docx_tool.py`:

```
python scripts/docx_tool.py extract  <file> <output_dir/> [--merge-runs] [--consolidate-revisions]
python scripts/docx_tool.py assemble <workspace/> <output.docx> [--original FILE] [--skip-verify]
python scripts/docx_tool.py verify   <path> [--original FILE] [--auto-fix] [--author NAME]
python scripts/docx_tool.py render   <file> [--range PAGES] [--dpi N] [--dest DIR]
python scripts/docx_tool.py convert  <file> --to FORMAT
python scripts/docx_tool.py accept-changes <input.docx> <output.docx>
python scripts/docx_tool.py annotate <workspace/> <id> "text" [--author NAME] [--reply-to ID]
```

---

## Quick Lookup

| Goal                   | Approach                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------- |
| Read/analyze content   | `pandoc` or `extract` for raw XML                                                       |
| Create new document    | JavaScript with `docx` library -- see [creation guide](references/creation-guide.md)    |
| Edit existing document | Extract -> edit XML -> assemble -- see [editing guide](references/xml-editing-guide.md) |
| Visual preview         | `render` command                                                                        |
| Validate structure     | `verify` command                                                                        |
| Accept all revisions   | `accept-changes` command                                                                |
| Add comments           | `annotate` command + manual markers                                                     |
| Convert legacy .doc    | `convert` command                                                                       |

---

## Environment Bootstrap (Run Before First Use)

Before executing any command, check and install missing dependencies. Run the appropriate commands silently and only report failures.

### macOS

```bash
# Python packages (always needed)
pip install defusedxml lxml 2>/dev/null || pip3 install defusedxml lxml

# System tools — install only what is missing
which pandoc  >/dev/null 2>&1 || brew install pandoc
which soffice >/dev/null 2>&1 || brew install --cask libreoffice
which pdftoppm >/dev/null 2>&1 || brew install poppler

# Node package (only needed for creating new documents from scratch)
# npm list -g docx >/dev/null 2>&1 || npm install -g docx
```

### Linux (Debian/Ubuntu)

```bash
pip install defusedxml lxml 2>/dev/null || pip3 install defusedxml lxml

which pandoc  >/dev/null 2>&1 || sudo apt-get install -y pandoc
which soffice >/dev/null 2>&1 || sudo apt-get install -y libreoffice
which pdftoppm >/dev/null 2>&1 || sudo apt-get install -y poppler-utils
```

### Dependency Matrix

| Dependency   | Required For                        | Install                           |
| ------------ | ----------------------------------- | --------------------------------- |
| `defusedxml` | extract, assemble, annotate, verify | `pip install defusedxml`          |
| `lxml`       | verify (XSD validation)             | `pip install lxml`                |
| `pandoc`     | text extraction to markdown         | `brew install pandoc`             |
| `soffice`    | render, convert, accept-changes     | `brew install --cask libreoffice` |
| `pdftoppm`   | render (page images)                | `brew install poppler`            |
| `docx` (npm) | creating new documents with JS      | `npm install -g docx`             |

**Policy:** When a command fails due to a missing tool, install it automatically and retry — do not ask the user for permission.
