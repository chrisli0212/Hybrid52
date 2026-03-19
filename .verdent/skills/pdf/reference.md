# PDF Reading — Advanced Reference

Supplementary material for advanced extraction, performance, and troubleshooting scenarios. Load this file only when the main SKILL.md guidance is insufficient.

## Table of Contents

- [pypdfium2 Text & Rendering](#pypdfium2-text--rendering)
- [Bounding-Box Text Export](#bounding-box-text-export)
- [pdfplumber Deep-Dive](#pdfplumber-deep-dive)
- [Batch Processing](#batch-processing)
- [Memory-Efficient Chunked Reading](#memory-efficient-chunked-reading)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Library Licenses](#library-licenses)

## pypdfium2 Text & Rendering

pypdfium2 wraps Chromium's PDFium engine. Fast rendering and reliable text output.

### Page Text

```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("document.pdf")
for i, page in enumerate(pdf):
    text = page.get_text()
    print(f"Page {i+1}: {len(text)} chars")
```

### High-Resolution Rendering

```python
import os

page = pdf[0]
bitmap = page.render(scale=3.0, rotation=0)
img = bitmap.to_pil()

img_dir = os.path.join(os.path.dirname("document.pdf") or ".", "images")
os.makedirs(img_dir, exist_ok=True)
img.save(os.path.join(img_dir, "hires.png"), "PNG")
```

### Multi-Page Render

```python
import os

img_dir = os.path.join(os.path.dirname("document.pdf") or ".", "images")
os.makedirs(img_dir, exist_ok=True)

for idx, page in enumerate(pdf):
    bitmap = page.render(scale=1.5)
    bitmap.to_pil().save(os.path.join(img_dir, f"page_{idx+1}.jpg"), "JPEG", quality=90)
```

## Bounding-Box Text Export

`pdftotext -bbox-layout` outputs XML with per-word coordinates — useful for spatial analysis or reconstructing layouts programmatically.

```bash
pdftotext -bbox-layout document.pdf output.xml
```

The XML contains `<word xMin yMin xMax yMax>` elements for each text fragment.

## pdfplumber Deep-Dive

### Character-Level Data

Every character on a page is available with position, font, and size:

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    page = pdf.pages[0]
    for ch in page.chars[:30]:
        print(f"'{ch['text']}' font={ch['fontname']} size={ch['size']:.1f} "
              f"x={ch['x0']:.1f} y={ch['top']:.1f}")
```

### Table Strategy Tuning

When default table detection fails:

```python
strategies = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_x_tolerance": 5,
    "snap_y_tolerance": 5,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
}
tables = page.extract_tables(strategies)
```

Strategy options:

- `"lines"` — use visible ruling lines
- `"text"` — infer structure from text alignment
- `"explicit"` — use manually supplied lines

### Visual Debug

Generate an image overlay showing detected table boundaries:

```python
import os

img = page.to_image(resolution=150)
img.debug_tablefinder()

img_dir = os.path.join(os.path.dirname("document.pdf") or ".", "images")
os.makedirs(img_dir, exist_ok=True)
img.save(os.path.join(img_dir, "debug_tables.png"))
```

## Batch Processing

Process a folder of PDFs with error isolation so one corrupt file does not abort the entire run.

```python
import os, glob, logging
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def batch_extract_text(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for pdf_path in glob.glob(os.path.join(input_dir, "*.pdf")):
        try:
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            out_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
            with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
                f.write("\n".join(text_parts))
            log.info("Done: %s", pdf_path)
        except Exception as exc:
            log.error("Failed: %s — %s", pdf_path, exc)
```

## Memory-Efficient Chunked Reading

For very large PDFs, avoid loading all pages at once. Process in fixed-size chunks:

```python
from pypdf import PdfReader

def chunked_read(pdf_path, chunk=20):
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        chunk_text = []
        for i in range(start, end):
            chunk_text.append(reader.pages[i].extract_text() or "")
        yield start, end, "\n".join(chunk_text)
```

Usage:

```python
for start, end, text in chunked_read("big.pdf"):
    print(f"Pages {start+1}-{end}: {len(text)} chars")
```

## Performance Notes

| Scenario                      | Recommendation                               |
| ----------------------------- | -------------------------------------------- |
| Plain text, speed priority    | `pdftotext` CLI is fastest                   |
| Structured tables             | pdfplumber with strategy tuning              |
| Page-to-image conversion      | pypdfium2 (lower memory than pdf2image)      |
| Embedded image extraction     | `pdfimages` CLI (much faster than rendering) |
| Scanned/image-only PDF        | OCR via pytesseract + pdf2image              |
| Very large files (>100 pages) | Chunked reading, 20 pages per batch          |

## Troubleshooting

### Empty text from extract_text()

Possible causes:

1. **Scanned PDF** — pages are images, not searchable text. Use OCR.
2. **Custom encoding** — try pdfplumber or pypdfium2 as alternatives.
3. **Encrypted** — decrypt first with `reader.decrypt("password")`.

### Garbled characters

Switch libraries. pypdfium2 handles CIDFont mappings better than pypdf in some documents.

### Table extraction misses columns

- Switch strategy from `"lines"` to `"text"` or vice versa.
- Adjust `snap_tolerance` and `intersection_tolerance`.
- Use `page.to_image().debug_tablefinder()` to visualize detection.

### Corrupt PDF

```bash
qpdf --check input.pdf
qpdf --replace-input input.pdf
```

## Library Licenses

| Library       | License      |
| ------------- | ------------ |
| pypdf         | BSD          |
| pdfplumber    | MIT          |
| pypdfium2     | Apache / BSD |
| pytesseract   | Apache       |
| pdf2image     | MIT          |
| poppler-utils | GPL-2        |
| qpdf          | Apache       |
