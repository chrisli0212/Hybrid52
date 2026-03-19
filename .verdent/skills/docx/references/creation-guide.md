# Generating New Word Documents with JavaScript

Build .docx files programmatically using the `docx` npm package, then validate the output.

## Installation

```bash
npm install -g docx
```

## Skeleton Structure

```javascript
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  ImageRun,
  Header,
  Footer,
  AlignmentType,
  PageOrientation,
  LevelFormat,
  ExternalHyperlink,
  TableOfContents,
  HeadingLevel,
  BorderStyle,
  WidthType,
  ShadingType,
  VerticalAlign,
  PageNumber,
  PageBreak,
} = require('docx');

const doc = new Document({
  sections: [
    {
      children: [
        /* paragraphs, tables, etc. */
      ],
    },
  ],
});
Packer.toBuffer(doc).then(buffer => fs.writeFileSync('output.docx', buffer));
```

After writing the file, always validate:

```bash
python scripts/docx_tool.py verify output.docx
```

## Page Dimensions

The library defaults to A4. Always set dimensions explicitly for predictable output.

Measurements use DXA units (twentieths of a point). 1 inch = 1440 DXA.

```javascript
sections: [
  {
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter: 8.5 x 11 inches
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      /* ... */
    ],
  },
];
```

| Paper     | Width (DXA) | Height (DXA) | Printable Width (1" margins) |
| --------- | ----------- | ------------ | ---------------------------- |
| US Letter | 12,240      | 15,840       | 9,360                        |
| A4        | 11,906      | 16,838       | 9,026                        |

**Landscape mode:** Supply portrait dimensions and set the orientation flag. The library handles the internal swap.

```javascript
size: {
  width: 12240,   // Short edge
  height: 15840,  // Long edge
  orientation: PageOrientation.LANDSCAPE
}
// Printable width becomes 15840 - left - right (uses the long dimension)
```

## Typography and Styles

Override the built-in heading styles by ID. Stick to widely available fonts like Arial or Calibri.

```javascript
const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Arial', size: 24 } } },
    paragraphStyles: [
      {
        id: 'Heading1',
        name: 'Heading 1',
        basedOn: 'Normal',
        next: 'Normal',
        quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial' },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2',
        name: 'Heading 2',
        basedOn: 'Normal',
        next: 'Normal',
        quickFormat: true,
        run: { size: 28, bold: true, font: 'Arial' },
        paragraph: { spacing: { before: 180, after: 180 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [
    {
      children: [
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          children: [new TextRun('Section Title')],
        }),
      ],
    },
  ],
});
```

Key details:

- Style IDs must match built-in names exactly ("Heading1", "Heading2", ...)
- `outlineLevel` is mandatory for Table of Contents integration (0 = H1, 1 = H2, ...)
- Font sizes are in half-points (24 = 12pt)

## Lists

Always use the numbering system. Never insert bullet characters manually.

```javascript
const doc = new Document({
  numbering: {
    config: [
      {
        reference: 'bullet-list',
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: '\u2022',
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
      {
        reference: 'ordered-list',
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: '%1.',
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      children: [
        new Paragraph({
          numbering: { reference: 'bullet-list', level: 0 },
          children: [new TextRun('First item')],
        }),
        new Paragraph({
          numbering: { reference: 'ordered-list', level: 0 },
          children: [new TextRun('Step one')],
        }),
      ],
    },
  ],
});
```

Numbering sequences: paragraphs sharing the same `reference` string form one continuous sequence. Use different references to restart numbering.

## Tables

Tables require width declarations at two levels: the table itself and each cell. Without both, rendering is inconsistent across Word, Google Docs, and LibreOffice.

```javascript
const cellBorder = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const allBorders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [4680, 4680],
  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders: allBorders,
          width: { size: 4680, type: WidthType.DXA },
          shading: { fill: 'D5E8F0', type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun('Content')] })],
        }),
      ],
    }),
  ],
});
```

Width rules:

- Always use `WidthType.DXA` (percentage widths break in Google Docs)
- `columnWidths` array must sum to the table `width`
- Each cell `width` must match its corresponding `columnWidths` entry
- Cell `margins` are internal padding and do not add to the cell width
- For full-width tables: width = page width - left margin - right margin
- Use `ShadingType.CLEAR` for backgrounds (SOLID produces black fills)

## Images

```javascript
new Paragraph({
  children: [
    new ImageRun({
      type: 'png', // Required: png, jpg, jpeg, gif, bmp, svg
      data: fs.readFileSync('image.png'),
      transformation: { width: 200, height: 150 },
      altText: { title: 'Title', description: 'Description', name: 'Name' },
    }),
  ],
});
```

The `type` parameter is mandatory. The `altText` object requires all three fields.

## Page Breaks

```javascript
// Standalone break
new Paragraph({ children: [new PageBreak()] });

// Break before a paragraph
new Paragraph({ pageBreakBefore: true, children: [new TextRun('New page')] });
```

A PageBreak must always be inside a Paragraph element.

## Table of Contents

```javascript
new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-3' });
```

Only works when headings use the `HeadingLevel` enum. Custom-styled headings will not appear in the TOC.

## Headers and Footers

```javascript
sections: [
  {
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({ children: [new TextRun('Document Title')] })],
      }),
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            children: [new TextRun('Page '), new TextRun({ children: [PageNumber.CURRENT] })],
          }),
        ],
      }),
    },
    children: [
      /* body content */
    ],
  },
];
```

## Common Mistakes to Avoid

- Forgetting to set page dimensions (defaults to A4, not US Letter)
- Using `\n` for line breaks (use separate Paragraph elements)
- Inserting bullet characters directly (use numbering config)
- Using `WidthType.PERCENTAGE` for tables (breaks Google Docs)
- Omitting cell widths (table + cell widths must both be set)
- Using `ShadingType.SOLID` (produces black backgrounds)
- Putting PageBreak outside a Paragraph
- Omitting `type` on ImageRun
- Using custom heading styles with Table of Contents (only HeadingLevel works)
- Forgetting `outlineLevel` on heading style definitions
