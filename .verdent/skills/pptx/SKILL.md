---
name: pptx
description: 'Presentation file toolkit for .pptx documents. Activate whenever the task involves creating, reading, editing, analyzing, or converting slide decks. Triggers on mentions of slides, decks, presentations, or .pptx filenames — whether the goal is content extraction, template-based modification, building new decks, or visual inspection.'
metadata:
  version: '1.1.0'
---

# Slide Deck Toolkit

**IMPORTANT: Script Path** — All bundled scripts live under this skill's own directory, NOT the user's project. The working directory at runtime is the user's project, so **relative paths like `scripts/deck.py` will fail**.

When this skill is loaded, the system message includes the skill's source path (e.g., `Source: /home/alice/.verdent/skills/pptx/SKILL.md`). Extract the directory portion and use it to build the absolute path to `scripts/deck.py`. For example, if the source is `/home/alice/.verdent/skills/pptx/SKILL.md`, then the script is at `/home/alice/.verdent/skills/pptx/scripts/deck.py`.

**Every invocation must use the full absolute path directly — do NOT use shell variables.** For example:

```bash
python3 /home/alice/.verdent/skills/pptx/scripts/deck.py read presentation.pptx
```

In the examples below, `DECK` is used as a **placeholder** for `python3 <skill_base_dir>/scripts/deck.py`. Replace it with the real absolute path in every actual command.

---

Determine what the user needs, then follow the matching path below.

**First time or unsure about dependencies?** Run the environment check first:

```bash
$DECK check-env
```

**Analyzing or extracting content?** Jump to [Ingestion](#ingestion).
**Modifying an existing deck or template?** Load [references/modify-workflow.md](references/modify-workflow.md).
**Building a new deck without a template?** Load [references/generate-with-js.md](references/generate-with-js.md).
**Need python-pptx native charts (column, pie, line, scatter)?** Load [references/python-charts.md](references/python-charts.md).

---

## Environment Check

Before starting any task, verify all required tools are installed:

```bash
$DECK check-env
```

This reports the status of all dependencies (Python packages, Node.js, PptxGenJS, LibreOffice, Poppler) and provides platform-specific install commands for any missing items.

If `render` is needed but LibreOffice is unavailable, inform the user and suggest installing it:

- **macOS**: `brew install --cask libreoffice`
- **Linux**: `sudo apt install libreoffice`
- **Windows**: Download from https://www.libreoffice.org

---

## Ingestion

### Readable Text Dump

```bash
$DECK read presentation.pptx
$DECK read presentation.pptx --format json
$DECK read presentation.pptx --skip-notes --out summary.txt
```

### Thumbnail Grid for Layout Analysis

```bash
$DECK thumbnails presentation.pptx
$DECK thumbnails presentation.pptx --columns 5
```

Produces a labeled JPEG grid showing each slide's XML filename. Hidden slides appear as hatched placeholders. Use this to identify which template layouts are available before starting edits.

### Raw XML Access

```bash
$DECK unpack presentation.pptx work/
```

Extracts the PPTX archive, formats every XML file for readability, and escapes smart quotes to XML entities so the Edit tool can safely modify content.

---

## Rendering for Visual QA

```bash
$DECK render output.pptx
$DECK render output.pptx --range 3-5 --dpi 200 --dest renders/
```

Converts slides to `slide-01.jpg`, `slide-02.jpg`, etc. via LibreOffice PDF export and Poppler rasterization. The runtime detects sandboxed environments automatically.

---

## Aesthetic Standards

Never settle for bland, text-heavy slides. Every deck should look intentionally designed.

### Choosing a Palette

Pick colors that connect to the subject matter. A tech startup pitch shouldn't share a palette with a healthcare report. One dominant hue (60-70%), one or two supporting tones, one sharp accent.

Some starting points:

- **Midnight Executive**: navy `1E2761`, ice `CADCFC`, white `FFFFFF`
- **Forest & Moss**: forest `2C5F2D`, moss `97BC62`, cream `F5F5F5`
- **Coral Energy**: coral `F96167`, gold `F9E795`, navy `2F3C7E`
- **Warm Terracotta**: terracotta `B85042`, sand `E7E8D1`, sage `A7BEAE`
- **Ocean Gradient**: deep `065A82`, teal `1C7293`, midnight `21295C`
- **Charcoal Minimal**: charcoal `36454F`, offwhite `F2F2F2`, black `212121`
- **Teal Trust**: teal `028090`, seafoam `00A896`, mint `02C39A`
- **Berry & Cream**: berry `6D2E46`, dusty rose `A26769`, cream `ECE2D0`
- **Sage Calm**: sage `84B59F`, eucalyptus `69A297`, slate `50808E`
- **Cherry Bold**: cherry `990011`, offwhite `FCF6F5`, navy `2F3C7E`

### Font Pairing

Headers need character; body text needs clarity. Don't default to Arial everywhere.

Recommended pairs — Georgia/Calibri, Arial Black/Arial, Cambria/Calibri, Trebuchet MS/Calibri, Palatino/Garamond.

Sizing: titles 36-44pt bold, section headers 20-24pt bold, body 14-16pt, captions 10-12pt muted.

### Avoiding Monotony

The most common failure mode is repeating one layout across every slide. Actively vary:

- Split-column (text + visual)
- Icon rows (colored circle background, bold label, description)
- 2x2 or 2x3 content grids
- Half-bleed image with text overlay
- Large stat callouts (60-72pt number, small caption below)
- Timeline or numbered-step process flows
- Quote/callout slides for testimonials or key messages
- Section dividers between major topics

Every slide needs at least one non-text element — image, chart, icon, or shape.

### Spacing Rules

- 0.5" minimum from slide edges
- 0.3-0.5" gaps between content blocks, consistent throughout
- Don't pack every inch — breathing room improves readability

### What Not to Do

- Repeating the same layout for consecutive slides
- Center-aligning body paragraphs (center titles only; left-align everything else)
- Weak contrast between title and body font sizes
- Defaulting to generic blue when the topic calls for something else
- Slides with only text and no visual anchor
- Forgetting text box padding when aligning shapes (set `margin: 0`)
- Low-contrast icon/text combinations against backgrounds
- Decorative underlines beneath titles (reads as machine-generated)

---

## Verification Protocol (Mandatory)

First attempts are almost never correct. Treat verification as a defect-hunting exercise.

### Step 1: Content Check

```bash
$DECK read output.pptx
$DECK read output.pptx | grep -iE "xxxx|lorem|placeholder|sample"
```

Confirm no missing content, wrong ordering, or leftover template placeholders.

### Step 2: Visual Inspection via Subagent

Render slides to images, then dispatch a subagent with fresh eyes:

```bash
$DECK render output.pptx
```

Subagent prompt:

```
These slide images likely contain layout issues. Find them.

Checklist:
- Overlapping or colliding elements (text through shapes, stacked layers)
- Text overflow, truncation, or excessive wrapping from narrow boxes
- Decorative elements displaced by text reflow
- Spacing < 0.3" between blocks or < 0.5" from slide edges
- Uneven whitespace distribution
- Column or grid misalignment
- Low-contrast text or icons against their background
- Leftover placeholder text

For every slide, report each issue with location and severity:
1. /path/to/slide-01.jpg (Expected: [what this slide should show])
2. /path/to/slide-02.jpg (Expected: ...)
```

### Step 3: Fix-and-Verify Loop

1. Generate -> Render -> Inspect
2. Catalogue all defects found
3. Fix each defect
4. Re-render only the affected slides, verify the fix didn't introduce new problems
5. Repeat until a full inspection pass finds zero new issues

Do not consider the task complete without at least one complete fix-verify cycle.

---

## Bundled Scripts (unified CLI)

All operations go through `scripts/deck.py`. Remember: `$DECK` below is a **placeholder** — always replace it with `python3 <skill_base_dir>/scripts/deck.py` using the real absolute path:

```
$DECK check-env
$DECK read    <file> [--format json] [--skip-notes] [--out FILE]
$DECK thumbnails <file> [--columns N]
$DECK unpack  <file> <workspace/>
$DECK pack    <workspace/> <output.pptx> [--original FILE]
$DECK render  <file> [--range PAGES] [--dpi N] [--dest DIR]
$DECK clone   <workspace/> <source.xml>
$DECK purge   <workspace/>
$DECK create  <spec.json> <output.pptx>
```

See [references/modify-workflow.md](references/modify-workflow.md) for the complete edit pipeline.

---

## Creating Decks from Scratch with JSON Spec

For programmatic deck creation with external images (e.g., matplotlib charts), use the `create` command with a JSON spec file:

```bash
$DECK create spec.json output.pptx
```

Example `spec.json`:

```json
{
  "width": 13.333,
  "height": 7.5,
  "slides": [
    {
      "background": "1E2761",
      "elements": [
        {
          "type": "text",
          "text": "Title Slide",
          "x": 1,
          "y": 2,
          "w": 11,
          "h": 2,
          "font_size": 44,
          "color": "D4A017",
          "bold": true,
          "align": "center"
        }
      ]
    },
    {
      "background": "F5F5F5",
      "elements": [
        {
          "type": "text",
          "text": "Chart Title",
          "x": 0.5,
          "y": 0.3,
          "w": 12,
          "h": 0.8,
          "font_size": 28,
          "color": "1E2761",
          "bold": true
        },
        { "type": "image", "path": "chart.png", "x": 1, "y": 1.3, "w": 11, "h": 5.5 }
      ]
    }
  ]
}
```

Supported element types:

- **text**: text, x, y, w, h, font_size, color, bold, align (left/center/right), font
- **image**: path (local file), x, y, w, h
- **shape**: shape (rectangle/rounded_rectangle/oval), x, y, w, h, fill, line

---

## Required Packages

**Python**: `pip install python-pptx Pillow defusedxml lxml`
**Node** (scratch builds only): `npm install -g pptxgenjs`
**System**: LibreOffice (`soffice`), Poppler (`pdftoppm`)
**Optional** (icons): `npm install -g react-icons react react-dom sharp`
