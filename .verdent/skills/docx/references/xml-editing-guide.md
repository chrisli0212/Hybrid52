# Editing Existing Word Documents via XML

Word documents are ZIP archives containing XML. To edit one, extract the archive,
modify the XML directly, and repack it. This guide covers the complete workflow.

## Three-Step Pipeline

### Step 1: Extract

```bash
python scripts/docx_tool.py extract document.docx workspace/
```

This unpacks the archive, formats XML for readability, merges fragmented text runs,
consolidates adjacent tracked changes from the same author, and escapes curly quotes
to XML entities for safe editing.

Use `--no-merge-runs` or `--no-consolidate-revisions` to skip those preprocessing steps.

### Step 2: Edit the XML

Work with files under `workspace/word/`. The main document body lives in `document.xml`.

**Use "Verdent" as the author name** for tracked changes and comments, unless a different name is explicitly requested.

**Prefer the Edit tool for direct string replacement.** Avoid writing Python scripts for
XML modifications — the Edit tool shows exactly what changes, reducing errors.

**Typographic quotes in new text:** When inserting content that includes apostrophes or
quotation marks, use XML entities for professional typography:

```xml
<w:t>Here&#x2019;s a quote: &#x201C;Hello&#x201D;</w:t>
```

| Entity     | Character                       |
| ---------- | ------------------------------- |
| `&#x2018;` | left single quote               |
| `&#x2019;` | right single quote / apostrophe |
| `&#x201C;` | left double quote               |
| `&#x201D;` | right double quote              |

**Adding comments:** Use the `annotate` command to set up the multi-file plumbing, then
place markers manually in document.xml:

```bash
python scripts/docx_tool.py annotate workspace/ 0 "Review this section"
python scripts/docx_tool.py annotate workspace/ 1 "Agreed, needs revision" --reply-to 0
```

Then add range markers (see Comments section below).

### Step 3: Reassemble

```bash
python scripts/docx_tool.py assemble workspace/ output.docx --original document.docx
```

This validates the XML with auto-repair, strips formatting whitespace, and creates
the final .docx. Use `--skip-verify` to bypass validation.

**Auto-repair handles:**

- durableId values exceeding OOXML limits (regenerates valid IDs)
- Missing `xml:space="preserve"` on text elements with leading/trailing whitespace

**Auto-repair does NOT handle:**

- Malformed XML, invalid element ordering, broken relationships, schema violations

---

## XML Patterns for Tracked Changes

### Inserting New Text

```xml
<w:ins w:id="1" w:author="Verdent" w:date="2025-01-01T00:00:00Z">
  <w:r><w:t>new text</w:t></w:r>
</w:ins>
```

### Deleting Existing Text

```xml
<w:del w:id="2" w:author="Verdent" w:date="2025-01-01T00:00:00Z">
  <w:r><w:delText>removed text</w:delText></w:r>
</w:del>
```

Inside `<w:del>`: always use `<w:delText>` instead of `<w:t>`, and `<w:delInstrText>` instead of `<w:instrText>`.

### Replacing Text (Minimal Diff)

Only mark the changed portion, not surrounding context:

```xml
<w:r><w:t>The deadline is </w:t></w:r>
<w:del w:id="1" w:author="Verdent" w:date="...">
  <w:r><w:delText>30</w:delText></w:r>
</w:del>
<w:ins w:id="2" w:author="Verdent" w:date="...">
  <w:r><w:t>60</w:t></w:r>
</w:ins>
<w:r><w:t> days.</w:t></w:r>
```

### Removing Entire Paragraphs

When deleting ALL content from a paragraph, also mark the paragraph break as deleted
so the empty line disappears when changes are accepted:

```xml
<w:p>
  <w:pPr>
    <w:numPr>...</w:numPr>
    <w:rPr>
      <w:del w:id="1" w:author="Verdent" w:date="2025-01-01T00:00:00Z"/>
    </w:rPr>
  </w:pPr>
  <w:del w:id="2" w:author="Verdent" w:date="2025-01-01T00:00:00Z">
    <w:r><w:delText>Content being removed...</w:delText></w:r>
  </w:del>
</w:p>
```

Without the `<w:del/>` inside `<w:pPr><w:rPr>`, an empty line remains after accepting.

### Rejecting Another Author's Insertion

Nest a deletion inside their insertion wrapper:

```xml
<w:ins w:author="OriginalAuthor" w:id="5">
  <w:del w:author="Verdent" w:id="10">
    <w:r><w:delText>their added text</w:delText></w:r>
  </w:del>
</w:ins>
```

### Restoring Another Author's Deletion

Add an insertion AFTER their deletion (do not modify the original `<w:del>`):

```xml
<w:del w:author="OriginalAuthor" w:id="5">
  <w:r><w:delText>text they removed</w:delText></w:r>
</w:del>
<w:ins w:author="Verdent" w:id="10">
  <w:r><w:t>text they removed</w:t></w:r>
</w:ins>
```

---

## Comments

After running `annotate` (Step 2), place range markers in document.xml.

**Markers are siblings of `<w:r>` elements, never nested inside them.**

### Standalone Comment

```xml
<w:commentRangeStart w:id="0"/>
<w:r><w:t>annotated text</w:t></w:r>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
```

### Threaded Reply

Nest the reply markers inside the parent's range:

```xml
<w:commentRangeStart w:id="0"/>
  <w:commentRangeStart w:id="1"/>
  <w:r><w:t>text with thread</w:t></w:r>
  <w:commentRangeEnd w:id="1"/>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="1"/></w:r>
```

---

## Embedding Images

1. Place the image file in `workspace/word/media/`
2. Add a relationship entry in `workspace/word/_rels/document.xml.rels`:
   ```xml
   <Relationship Id="rId5" Type=".../image" Target="media/image1.png"/>
   ```
3. Declare the content type in `workspace/[Content_Types].xml`:
   ```xml
   <Default Extension="png" ContentType="image/png"/>
   ```
4. Reference in document.xml:
   ```xml
   <w:drawing>
     <wp:inline>
       <wp:extent cx="914400" cy="914400"/>  <!-- EMUs: 914400 = 1 inch -->
       <a:graphic>
         <a:graphicData uri=".../picture">
           <pic:pic>
             <pic:blipFill><a:blip r:embed="rId5"/></pic:blipFill>
           </pic:pic>
         </a:graphicData>
       </a:graphic>
     </wp:inline>
   </w:drawing>
   ```

---

## XML Schema Notes

### Element Ordering in `<w:pPr>`

Elements must appear in this sequence: `<w:pStyle>`, `<w:numPr>`, `<w:spacing>`, `<w:ind>`, `<w:jc>`, `<w:rPr>` last.

### Whitespace Handling

Any `<w:t>` element with leading or trailing spaces requires `xml:space="preserve"`.

### Revision Session IDs

RSIDs must be 8-digit hexadecimal values (e.g., `00AB1234`).

---

## Editing Best Practices

- **Replace whole `<w:r>` blocks** when adding tracked changes. Don't inject change markers inside a run.
- **Preserve `<w:rPr>` formatting**: Copy the original run's formatting properties into tracked-change runs.
- **Keep edits minimal**: Only wrap the actual changed text in change markers, not surrounding content.
- **Unique IDs**: Every `w:id` attribute within `<w:ins>`, `<w:del>`, comment markers, and bookmarks must be unique within the file.
