# Modifying Existing Decks

This workflow applies when you have a `.pptx` file (a template or a prior version) and need to change its content.

## Pipeline Overview

```
Inspect -> Plan -> Unpack -> Restructure -> Fill Content -> Purge Orphans -> Repack
```

### 1. Inspect

Get a visual map of what the template offers:

```bash
$DECK thumbnails template.pptx
$DECK read template.pptx
```

Review the thumbnail grid to see available layouts. Review extracted text to see what placeholder strings exist.

### 2. Plan Layout Mapping

For each section of your target content, pick the best-matching template slide. **Prioritize visual variety** — don't reuse the same layout repeatedly. Match content type to layout:

- Key arguments -> bullet layout
- Team/people info -> multi-column grid
- Metrics/KPIs -> stat callout layout
- Testimonials -> quote slide
- Transitions -> section divider

If the template has more layout types than you initially planned to use, reconsider your content structure. Using diverse layouts is almost always worth it.

### 3. Unpack

```bash
$DECK unpack template.pptx work/
```

This extracts the ZIP, pretty-prints XML, and converts typographic quotes to XML entities.

### 4. Restructure (do this yourself, not subagents)

Slide order is controlled by `ppt/presentation.xml` > `<p:sldIdLst>`.

**Delete slides**: Remove the `<p:sldId>` entry. Run `purge` later to clean up files.

**Reorder slides**: Rearrange `<p:sldId>` elements within `<p:sldIdLst>`.

**Clone a slide for reuse**:

```bash
$DECK clone work/ slide2.xml        # duplicate an existing slide
$DECK clone work/ slideLayout3.xml   # new slide from a layout
```

The script prints a `<p:sldId>` element — insert it at your desired position in `<p:sldIdLst>`.

**Finish all structural changes before proceeding to content.**

### 5. Fill Content

Each slide is an independent XML file in `ppt/slides/`, so **subagents can work on multiple slides in parallel**. When delegating to subagents, always include:

- The target slide path(s)
- **"Use the Edit tool for all modifications"**
- The formatting rules and gotchas from this document

For each slide file:

1. Read the XML
2. Locate every placeholder — text, images, charts, icons, captions
3. Replace with final content

**Always use the Edit tool**, not sed or scripts. The Edit tool forces precision about what changes.

#### XML Formatting Rules

**Bold treatment**: Apply `b="1"` on `<a:rPr>` for all headings, subheadings, and inline labels (e.g. "Status:", "Description:" at line start).

**List formatting**: Never insert Unicode bullet characters (`U+2022`). Use `<a:buChar>` or `<a:buAutoNum>` for proper bullets. Let bullet style inherit from the layout unless overriding.

**Separate paragraphs for separate items**: Multiple list items or numbered steps must each be their own `<a:p>` element. Never concatenate them into a single text run.

Bad — everything jammed into one paragraph:

```xml
<a:p>
  <a:r><a:rPr .../><a:t>Step 1: Do X. Step 2: Do Y.</a:t></a:r>
</a:p>
```

Good — each item gets its own paragraph, with `<a:pPr>` copied from the original to preserve spacing:

```xml
<a:p>
  <a:pPr algn="l"><a:lnSpc><a:spcPts val="3919"/></a:lnSpc></a:pPr>
  <a:r><a:rPr lang="en-US" sz="2799" b="1" .../><a:t>Step 1</a:t></a:r>
</a:p>
<a:p>
  <a:pPr algn="l"><a:lnSpc><a:spcPts val="3919"/></a:lnSpc></a:pPr>
  <a:r><a:rPr lang="en-US" sz="2799" .../><a:t>Do X.</a:t></a:r>
</a:p>
```

#### Handling Mismatched Counts

When your content has fewer items than the template expects (e.g. template shows 4 team members, you have 3):

- Delete the entire excess group (image + text boxes + shapes), not just the text
- Run visual QA after removal to catch layout gaps

When replacement text is significantly longer:

- May overflow or wrap unexpectedly
- Always verify with visual QA
- Consider trimming content to respect the layout's constraints

#### Typographic Quotes

The unpack/pack cycle handles quote normalization. But the Edit tool converts smart quotes to plain ASCII. When inserting text that contains quotation marks, use XML entities:

```xml
<a:t>the &#x201C;Agreement&#x201D;</a:t>
```

| Glyph  | Name               | Entity     |
| ------ | ------------------ | ---------- |
| \u201c | Left double quote  | `&#x201C;` |
| \u201d | Right double quote | `&#x201D;` |
| \u2018 | Left single quote  | `&#x2018;` |
| \u2019 | Right single quote | `&#x2019;` |

#### Whitespace and Namespace Safety

- Add `xml:space="preserve"` to `<a:t>` elements with leading or trailing spaces
- Parse XML with `defusedxml.minidom` — `xml.etree.ElementTree` corrupts namespace declarations

### 6. Purge Orphans

```bash
$DECK purge work/
```

Removes slides absent from `<p:sldIdLst>`, unreferenced media/charts/diagrams, orphaned relationship files, and stale Content_Types entries.

### 7. Repack

```bash
$DECK pack work/ output.pptx --original template.pptx
```

When `--original` is provided, the packer runs structural integrity checks (well-formed XML, ID uniqueness, relationship consistency, content type completeness) and auto-repairs minor issues like missing `xml:space="preserve"`. XML whitespace is compressed before archiving.

### After Repacking

Proceed to the **Verification Protocol** in SKILL.md. Always render and inspect before declaring success.
