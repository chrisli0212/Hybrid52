---
name: pdf
description: PDF toolkit for reading, analyzing, and creating PDF documents. Extracts text, tables, images, LaTeX formulas, and metadata from PDF files with page-range control. Creates new PDFs via reportlab (programmatic/structured) or LaTeX/Tectonic (academic). Use when the agent needs to read, parse, analyze, or generate PDF content.
metadata:
  version: '3.0.0'
---

# PDF Content Extraction Guide

## File Writing Rule

When you need to write text content to a file (e.g. `.tex`, `.py`, `.txt`, `.csv`), **always use the `file_write` tool** instead of bash commands. Never pass large text blobs through bash (no heredocs, no `echo >`, no `cat <<EOF`, no `python -c "open(...).write(...)"`). Write the file with `file_write` first, then run any compilation or processing commands via bash separately.

## No Subagent for PDF Reading

**Do NOT spawn subagents to read PDFs or extract content.** Subagents cannot effectively return the extracted text/context back to you — the content will be lost or truncated. Always read and extract PDF content yourself directly using the tools and code in this guide. If the PDF is very long, split it into batches (up to 8 pages per batch), read and process each batch yourself sequentially.

## Page Range Convention

All extraction operations support a **page range** parameter to target specific pages.

Format: `"1-5"`, `"3"`, `"10-20"`, `"1,3,7-9"`

- Single page: `"4"` (page 4 only)
- Continuous range: `"2-8"` (pages 2 through 8)
- Mixed: `"1,3,5-10"` (pages 1, 3, and 5 through 10)

**Limit: process at most 8 pages per request.** If the target range exceeds 8 pages, split into multiple batches and process sequentially.

Helper to resolve a range string into a page index list (0-based):

```python
def parse_page_range(range_str, total_pages):
    pages = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(int(end), total_pages)
            pages.update(range(start, end + 1))
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                pages.add(p)
    result = sorted(pages)
    if len(result) > 8:
        raise ValueError(f"Range covers {len(result)} pages; max 8 per request. Split into batches.")
    return [p - 1 for p in result]  # convert to 0-based
```

## Text Extraction

### Basic — pypdf

```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")
page_indices = parse_page_range("1-5", len(reader.pages))
for idx in page_indices:
    print(reader.pages[idx].extract_text())
```

### Layout-Preserving — pdfplumber

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    page_indices = parse_page_range("1-5", len(pdf.pages))
    for idx in page_indices:
        text = pdf.pages[idx].extract_text()
        print(text)
```

### Character-Level Coordinates — pdfplumber

```python
with pdfplumber.open("document.pdf") as pdf:
    page = pdf.pages[0]
    for ch in page.chars[:20]:
        print(f"'{ch['text']}' x={ch['x0']:.1f} y={ch['y0']:.1f}")
```

### Region Extraction

Extract text from a specific rectangular area on a page:

```python
with pdfplumber.open("document.pdf") as pdf:
    page = pdf.pages[0]
    region = page.within_bbox((left, top, right, bottom))
    print(region.extract_text())
```

### Command-Line — pdftotext (poppler-utils)

```bash
# Full text
pdftotext input.pdf output.txt

# Preserve spatial layout
pdftotext -layout input.pdf output.txt

# Specific page range (pages 3 to 7)
pdftotext -f 3 -l 7 input.pdf output.txt
```

## Table Extraction

### PyMuPDF (Preferred)

PyMuPDF (`fitz`) provides the most accurate table detection and extraction:

```python
import fitz

doc = fitz.open("report.pdf")
page_indices = parse_page_range("1-5", len(doc))
for idx in page_indices:
    page = doc[idx]
    tabs = page.find_tables()
    for table in tabs.tables:
        df = table.to_pandas()
        print(df)
```

### Fallback — pdfplumber

If PyMuPDF is unavailable or misses tables, use pdfplumber:

```python
import pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    page_indices = parse_page_range("1-5", len(pdf.pages))
    for idx in page_indices:
        tables = pdf.pages[idx].extract_tables()
        for table in tables:
            for row in table:
                print(row)
```

Tune detection strategies when default extraction misses rows or merges columns:

```python
settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "intersection_tolerance": 15,
}
tables = page.extract_tables(settings)
```

### Export Tables to DataFrame

```python
import pandas as pd
import fitz

doc = fitz.open("data.pdf")
frames = []
page_indices = parse_page_range("1-8", len(doc))
for idx in page_indices:
    page = doc[idx]
    tabs = page.find_tables()
    for table in tabs.tables:
        frames.append(table.to_pandas())
if frames:
    combined = pd.concat(frames, ignore_index=True)
```

## Image & Visual Extraction

### Extract Embedded Images — pdfimages (Preferred)

**Always try `pdfimages` first** to extract images embedded in the PDF. This extracts the original image data without quality loss and is much faster than rendering pages.

```bash
# List all embedded images with details
pdfimages -list document.pdf

# Extract in original format (preserves PNG/JPEG/etc.)
pdfimages -all document.pdf output_prefix

# Extract as JPEG
pdfimages -j document.pdf output_prefix
```

### Convert to High-Res PNG — CLI

```bash
# All pages at 300 DPI
pdftoppm -png -r 300 document.pdf output_prefix

# Pages 2-4 at 600 DPI
pdftoppm -png -r 600 -f 2 -l 4 document.pdf output_prefix
```

### Fallback: Render Pages as Images (for scanned PDFs)

If the PDF is scanned (pages are full-page images, not searchable text), `pdfimages` may not help. In that case, render pages to images as a fallback:

#### pypdfium2

```python
import pypdfium2 as pdfium
import os
from PIL import Image

pdf = pdfium.PdfDocument("document.pdf")
page_indices = parse_page_range("1-3", len(pdf))

img_dir = os.path.join(os.path.dirname("document.pdf") or ".", "images")
os.makedirs(img_dir, exist_ok=True)

for idx in page_indices:
    bitmap = pdf[idx].render(scale=2.0)
    pil_img = bitmap.to_pil()
    pil_img.save(os.path.join(img_dir, f"page_{idx + 1}.png"))
```

#### pdf2image

```python
from pdf2image import convert_from_path
import os

images = convert_from_path("document.pdf", first_page=1, last_page=5, dpi=200)

img_dir = os.path.join(os.path.dirname("document.pdf") or ".", "images")
os.makedirs(img_dir, exist_ok=True)

for i, img in enumerate(images, start=1):
    img.save(os.path.join(img_dir, f"page_{i}.png"), "PNG")
```

## Metadata Inspection

```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")
meta = reader.metadata
print(f"Title:    {meta.title}")
print(f"Author:   {meta.author}")
print(f"Subject:  {meta.subject}")
print(f"Creator:  {meta.creator}")
print(f"Pages:    {len(reader.pages)}")
```

## OCR for Scanned Documents

When `extract_text()` returns empty or garbled results, fall back to OCR:

```python
from pdf2image import convert_from_path
import pytesseract

images = convert_from_path("scanned.pdf", first_page=1, last_page=5)
for i, img in enumerate(images, start=1):
    text = pytesseract.image_to_string(img)
    print(f"--- Page {i} ---")
    print(text)
```

## LaTeX Formula Extraction & Rendering

PDF documents from academic papers, textbooks, and technical reports frequently contain mathematical formulas. **Every LaTeX formula must be extracted and faithfully reproduced — never skip, simplify, or approximate any formula.** Treat formula extraction with the same priority as text extraction.

**Do not forget tables.** Academic PDFs often contain large tables with data, comparisons, or results. When extracting content from a LaTeX-generated PDF, always check for tables on every page using `extract_tables()` and convert them to proper LaTeX `tabular`/`table` environments. Large tables spanning multiple columns or pages are easily overlooked — treat table extraction with the same priority as formula extraction.

### Default: Text-Based Formula Extraction

This is the default method. Most academic PDFs (especially LaTeX/arXiv-generated) have readable math content via `extract_text()`.

1. **Detect PDF type:**

   ```python
   from pypdf import PdfReader
   reader = PdfReader("paper.pdf")
   creator = (reader.metadata.creator or "").lower()
   is_text_native = any(k in creator for k in ["tex", "latex", "pdftex", "arxiv"])
   ```

2. **Extract formulas via text patterns:**
   - Lines with high density of Unicode math symbols (∑, ∫, ∂, ∈, ∪, etc.)
   - Equation numbering patterns like `.(N)` at end of line
   - Known LaTeX-output patterns: `πθ` → `\pi_\theta`, `∼` → `\sim`

3. **Reconstruct LaTeX manually:** The text preserves semantic content but loses spatial structure (fractions become flat, subscripts/superscripts disappear). Reconstruct LaTeX by mapping extracted text back to standard LaTeX notation.

**When to use:** `extract_text()` returns readable math symbols. This covers most LaTeX-generated PDFs.

**When NOT to use:** Scanned PDFs, image-heavy PDFs, or when `extract_text()` returns empty/garbled output — fall back to image-based methods below.

### Fallback: Image-Based Formula Extraction

When text extraction fails (garbled, empty, or scanned PDF), use image-based methods. **Preferred approach:** Use font-based detection to locate formula regions precisely, then crop and recognize.

#### Option A: Font-Based Detection + pix2tex (best precision)

Use pdfplumber's character-level font data to find math-font clusters, crop tightly, then recognize with pix2tex:

```python
import pdfplumber, pypdfium2 as pdfium, re
from pix2tex.cli import LatexOCR

MATH_FONT_PATTERN = re.compile(
    r"(CMSY|CMMI|CMEX|CMSS|CMR|Math|Symbol|MT Extra|Cambria.*Math|STIX|rsfs|msbm|msam|eufm|bbold)",
    re.IGNORECASE
)

def find_formula_regions(page, merge_gap=8):
    math_chars = [
        ch for ch in page.chars
        if MATH_FONT_PATTERN.search(ch.get("fontname", ""))
    ]
    if not math_chars:
        return []
    clusters, current = [], [math_chars[0]]
    for ch in math_chars[1:]:
        prev = current[-1]
        if abs(ch["top"] - prev["top"]) < merge_gap and ch["x0"] - prev["x1"] < merge_gap * 3:
            current.append(ch)
        else:
            clusters.append(current)
            current = [ch]
    clusters.append(current)
    regions = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        regions.append((
            min(c["x0"] for c in cluster), min(c["top"] for c in cluster),
            max(c["x1"] for c in cluster), max(c["bottom"] for c in cluster),
        ))
    regions.sort(key=lambda r: (r[1], r[0]))
    merged = [list(regions[0])] if regions else []
    for x0, top, x1, bottom in regions[1:]:
        p = merged[-1]
        if top <= p[3] + merge_gap and x0 <= p[2] + merge_gap * 3:
            p[0], p[1], p[2], p[3] = min(p[0], x0), min(p[1], top), max(p[2], x1), max(p[3], bottom)
        else:
            merged.append([x0, top, x1, bottom])
    return merged
```

Then render + crop + recognize:

```python
RENDER_SCALE, PADDING_PX = 3.0, 6
model = LatexOCR()
pdf_doc = pdfium.PdfDocument("document.pdf")

with pdfplumber.open("document.pdf") as pdf:
    page = pdf.pages[0]
    regions = find_formula_regions(page)
    bitmap = pdf_doc[0].render(scale=RENDER_SCALE)
    full_img = bitmap.to_pil()
    sx, sy = full_img.size[0] / float(page.width), full_img.size[1] / float(page.height)
    img_w, img_h = full_img.size
    for i, (x0, top, x1, bottom) in enumerate(regions):
        crop_box = (
            max(0, int(x0 * sx) - PADDING_PX), max(0, int(top * sy) - PADDING_PX),
            min(img_w, int(x1 * sx) + PADDING_PX), min(img_h, int(bottom * sy) + PADDING_PX),
        )
        latex_str = model(full_img.crop(crop_box))
        print(f"Formula {i+1}: {latex_str}")
```

#### Option B: Nougat (full-page, no region detection needed)

[Nougat](https://github.com/facebookresearch/nougat) converts entire academic PDF pages to Markdown with LaTeX math. Use when font-based detection returns nothing or for bulk extraction:

```bash
nougat paper.pdf -o output_dir --no-skipping
```

The output `.mmd` file contains inline `$...$` and display `$$...$$` LaTeX formulas already parsed.

### Generating Academic PDF via LaTeX

For academic or math-heavy documents, compose the entire document — text and formulas together — into a single `.tex` file, then compile once to produce a publication-quality PDF. **Do NOT render formulas in isolation.**

**Limit: process at most 8 pages worth of content per batch.** If the document exceeds 8 pages, split into multiple batches and write incrementally.

**Core workflow:**

1. **Prepare all content** (text paragraphs, section headings, formulas, captions, tables, etc.) — up to 8 pages per batch
2. **Write a `.tex` document** — plain text goes as normal LaTeX, formulas use LaTeX notation (inline `$...$` or display `\begin{equation}...\end{equation}`)
3. **Compile once** via Tectonic to produce the final PDF

This produces a real academic-grade PDF with unified fonts, perfect formula baseline alignment, and no quality loss.

#### Install Tectonic

Tectonic is a lightweight, modern LaTeX engine that auto-resolves packages at first compile. No sudo/admin password required — it downloads a single binary.

**Auto-install if not present** (agent should do this automatically):

```python
import subprocess, shutil, platform, os

def ensure_tectonic():
    if shutil.which("tectonic"):
        return True
    system = platform.system()
    if system in ("Darwin", "Linux"):
        subprocess.run(
            ["sh", "-c", "curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh"],
            check=True
        )
        # Move to a PATH directory
        if os.path.exists("./tectonic"):
            dest = os.path.expanduser("~/.local/bin/tectonic")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.rename("./tectonic", dest)
    elif system == "Windows":
        subprocess.run(
            ["powershell", "-Command",
             "[System.Net.ServicePointManager]::SecurityProtocol = "
             "[System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
             "iex ((New-Object System.Net.WebClient).DownloadString('https://drop-ps1.fullyjustified.net'))"],
            check=True
        )
    return shutil.which("tectonic") is not None
```

#### Compile Full Document to PDF

Tectonic supports **all** LaTeX syntax — `cases`, `align`, `array`, matrices, `\text{}`, nested delimiters, etc. Packages are downloaded automatically on first use.

**Workflow:** Use `file_write` to create the `.tex` file, then run `tectonic` via bash to compile it. Do NOT write `.tex` content through bash.

**Incremental writing rule: NEVER write the entire `.tex` file in one shot.** For documents with multiple sections, follow this process:

1. **First pass:** Use `file_write` to create the `.tex` file with the preamble (`\documentclass` through `\begin{document}`) and the first 1-2 sections, ending with `\end{document}`.
2. **Extract figures and tables:** Extract embedded images with `pdfimages -all`. For tables captured as screenshots, **MUST tight-crop to the table's exact boundary** (no surrounding text, no headers/footers, no page margins) — verify each cropped image before proceeding. Save all images to the `images/` subdirectory.
3. **Read source for next batch:** Read the next few pages to extract the next section(s).
4. **Append sections:** Use `file_edit` to insert the new section(s) before `\end{document}`. Reference figures/tables with `\includegraphics` and `\begin{figure}`/`\begin{table}` environments.
5. **Repeat** steps 2-4 until all sections are written.
6. **Compile once** after all sections are in place: `tectonic /path/to/main.tex`

This prevents context overload and ensures each section is carefully extracted and written. Typical batch size: 1-2 sections or ~2-4 pages of source content per pass.

```
Step 1: file_write main.tex — preamble + first 1-2 sections + \end{document}
Step 2: pdfimages for figures; tight-crop Table screenshots to exact boundary; save to images/
Step 3: Read next pages from source
Step 4: file_edit main.tex — insert next section(s) + \includegraphics references before \end{document}
Step 5: Repeat steps 2-4 for remaining sections
Step 6: Run bash: tectonic /path/to/main.tex
```

**Example: recomposing an extracted paper into LaTeX:**

First, use `file_write` to create the `.tex` file:

```latex
\documentclass[12pt]{article}
\usepackage[a4paper, margin=2cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}

\title{Extracted Paper Title}
\author{Original Authors}
\date{}

\begin{document}
\maketitle

\section{Introduction}

We consider the optimization problem where the objective is defined as:

\begin{equation}
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log p_\theta(x) \right]
\end{equation}

The policy gradient is computed via:

\begin{align}
\nabla_\theta \mathcal{L} &= \sum_{i=1}^{N} \nabla_\theta \log \pi_\theta(a_i | s_i) \cdot A_i
\end{align}

The reward function is:

\begin{equation}
r(s, a) = \begin{cases}
r_{\text{task}} & \text{if task completed} \\
0 & \text{otherwise}
\end{cases}
\end{equation}

\end{document}
```

Then compile via bash:

```bash
tectonic /path/to/main.tex
```

### Verification Checklist

After extracting formulas from a PDF, always verify:

1. **Completeness** — Count formulas in the source PDF (visual scan) vs. extracted count. **Zero tolerance for missed formulas.**
2. **Structural fidelity** — Fractions, subscripts, superscripts, matrices, and nested expressions must be structurally correct.
3. **Round-trip rendering** — Render each extracted LaTeX string back to an image and visually compare with the original PDF.
4. **Edge cases to watch:**
   - Multi-line equations (`align`, `gather`, `cases` environments)
   - Inline vs. display math mode
   - Equation numbering
   - Special operators (`\operatorname`, `\mathbb`, `\mathcal`)
   - Matrices and arrays

## PDF Creation (Non-LaTeX)

For academic/math-heavy documents, use the LaTeX + Tectonic workflow above. For all other PDF creation needs, use **reportlab**.

**Decision flow:**

- Non-academic PDF (invoices, reports, certificates, data-driven documents)? → **reportlab**
- Academic paper with math formulas? → **LaTeX + Tectonic** (see above)

### reportlab — Programmatic / Structured PDFs

Best for: invoices, certificates, reports with charts, data-driven documents, form-like layouts.

#### Basic Document

```python
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import cm, mm, inch
from reportlab.pdfgen import canvas

c = canvas.Canvas("output.pdf", pagesize=A4)
width, height = A4

c.setFont("Helvetica-Bold", 18)
c.drawString(2*cm, height - 3*cm, "Report Title")

c.setFont("Helvetica", 11)
c.drawString(2*cm, height - 5*cm, "Generated on 2025-01-15")

c.showPage()
c.save()
```

#### Platypus — High-Level Layout Engine

For multi-page documents with automatic pagination, headers, and flowing content, use Platypus (Page Layout and Typography Using Scripts):

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

doc = SimpleDocTemplate("report.pdf", pagesize=A4,
                        topMargin=2*cm, bottomMargin=2*cm,
                        leftMargin=2.5*cm, rightMargin=2.5*cm)

styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Quarterly Report", styles["Title"]))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("This is the executive summary paragraph.", styles["BodyText"]))
story.append(Spacer(1, 1*cm))

data = [
    ["Item", "Q1", "Q2", "Q3"],
    ["Revenue", "$10M", "$12M", "$15M"],
    ["Costs", "$8M", "$9M", "$10M"],
]
table = Table(data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#D9E2F3")]),
]))
story.append(table)

doc.build(story)
```

#### Subscripts and Superscripts

reportlab uses XML-style markup — **never use Unicode superscript/subscript characters** (they render incorrectly in most PDF fonts):

```python
# Correct — XML tags
Paragraph("H<sub>2</sub>O and x<sup>2</sup>", styles["BodyText"])

# WRONG — Unicode characters (will render as missing glyphs)
# Paragraph("H₂O and x²", styles["BodyText"])
```

#### Drawing Graphics

```python
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String
from reportlab.graphics import renderPDF

d = Drawing(400, 200)
d.add(Rect(50, 50, 100, 80, fillColor=colors.lightblue, strokeColor=colors.black))
d.add(Circle(250, 100, 40, fillColor=colors.salmon))
d.add(String(50, 160, "Chart Title", fontSize=14, fontName="Helvetica-Bold"))
renderPDF.drawToFile(d, "drawing.pdf", "Graphics")
```

#### Images in reportlab

```python
from reportlab.platypus import Image

story.append(Image("chart.png", width=12*cm, height=8*cm))
```

### Non-ASCII / CJK Font Handling

reportlab requires explicit font configuration for non-Latin text. See the "Encoding & Font Pitfalls" section below for font registration details.

## Handling Encrypted PDFs

```python
from pypdf import PdfReader

reader = PdfReader("protected.pdf")
if reader.is_encrypted:
    reader.decrypt("password")
text = reader.pages[0].extract_text()
```

## Encoding & Font Pitfalls When Writing PDFs

When generating or annotating PDFs that contain non-ASCII text (CJK, Arabic, Cyrillic, accented Latin, symbols, etc.), follow these rules to avoid garbled output:

1. **Use a Unicode-capable font.** Built-in PDF fonts (Helvetica, Times-Roman, Courier) only cover Latin-1. For any text outside that range, register and embed a TrueType/OpenType font that covers the required glyphs.

   reportlab example:

   ```python
   from reportlab.pdfbase import pdfmetrics
   from reportlab.pdfbase.ttfonts import TTFont

   pdfmetrics.registerFont(TTFont("MyFont", "/path/to/font.ttf"))
   canvas.setFont("MyFont", 12)
   ```

2. **Verify glyph coverage before rendering.** A font may support a language's main characters but miss specific symbols (e.g., bullet `U+2022`, em-dash `U+2014`, curly quotes). If a glyph is missing the PDF renderer may silently substitute a wrong character or show a blank. Pre-check with:

   ```python
   from fontTools.ttLib import TTFont as FTFont

   font = FTFont("/path/to/font.ttf")
   cmap = font.getBestCmap()
   missing = [ch for ch in text if ord(ch) not in cmap]
   ```

   Replace or strip any characters not in the font's cmap before writing.

3. **Normalize Unicode before embedding.** Use NFC normalization so that composed and decomposed forms are consistent:

   ```python
   import unicodedata
   text = unicodedata.normalize("NFC", text)
   ```

4. **Sanitize special symbols.** Common culprits that cause garbled output:
   - Bullet `U+2022` — replace with `U+00B7` (middle dot) or a hyphen if unsupported
   - Smart quotes `U+201C` `U+201D` — fall back to straight quotes `"`
   - Em-dash `U+2014` — fall back to `--`
   - Ellipsis `U+2026` — fall back to `...`

5. **Always embed fonts.** Set `embeddedFont=True` or equivalent so the PDF is self-contained and renders correctly on any machine, regardless of locally installed fonts.

6. **Test with a round-trip.** After generating, re-read the PDF with pypdf or pdfplumber and compare extracted text against the original to catch silent corruption.

## Advanced Topics

For additional extraction techniques, performance tuning, and troubleshooting:

- **pypdfium2 text extraction and rendering**: See [reference.md](reference.md)
- **Bounding-box text export (pdftotext -bbox-layout)**: See [reference.md](reference.md)
- **Batch processing large document sets**: See [reference.md](reference.md)
- **Memory-efficient chunked reading**: See [reference.md](reference.md)

## Quick Lookup

| Goal                                      | Recommended Tool             | Key API                                                 |
| ----------------------------------------- | ---------------------------- | ------------------------------------------------------- |
| Plain text                                | pypdf                        | `page.extract_text()`                                   |
| Layout-aware text                         | pdfplumber                   | `page.extract_text()`                                   |
| Tables                                    | PyMuPDF (fitz)               | `page.find_tables()` → `table.to_pandas()`              |
| Page images                               | pypdfium2 / pdf2image        | `render()` / `convert_from_path()`                      |
| Embedded images                           | pdfimages (CLI)              | `pdfimages -all`                                        |
| Metadata                                  | pypdf                        | `reader.metadata`                                       |
| OCR (scanned)                             | pytesseract + pdf2image      | `image_to_string()`                                     |
| LaTeX formulas (text-native)              | pypdf + text patterns        | Method 0: `extract_text()` + pattern matching           |
| LaTeX formulas (scanned/complex)          | nougat / pix2tex (LaTeX-OCR) | `nougat` CLI / `LatexOCR()`                             |
| Recompose full PDF                        | Tectonic (local)             | `tectonic main.tex` — auto-install, full syntax support |
| Create structured PDF (invoices, reports) | reportlab                    | `SimpleDocTemplate` + Platypus `story`                  |
| CLI text extract                          | pdftotext                    | `pdftotext -f -l`                                       |
