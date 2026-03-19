# Generating Decks with PptxGenJS

When no template exists, build presentations programmatically with PptxGenJS in Node.js.

## Minimal Example

```javascript
const pptxgen = require('pptxgenjs');

let deck = new pptxgen();
deck.layout = 'LAYOUT_16x9';
deck.author = 'Author';
deck.title = 'Title';

let s = deck.addSlide();
s.addText('Hello', { x: 0.5, y: 0.5, fontSize: 36, color: '363636' });

deck.writeFile({ fileName: 'output.pptx' });
```

## Canvas Sizes

All coordinates are inches. Available presets:

- `LAYOUT_16x9` — 10 x 5.625 (default)
- `LAYOUT_16x10` — 10 x 6.25
- `LAYOUT_4x3` — 10 x 7.5
- `LAYOUT_WIDE` — 13.3 x 7.5

---

## Working with Text

### Placement and Styling

```javascript
s.addText('Title', {
  x: 1,
  y: 1,
  w: 8,
  h: 2,
  fontSize: 24,
  fontFace: 'Arial',
  color: '363636',
  bold: true,
  align: 'center',
  valign: 'middle',
});
```

### Character Spacing

Use `charSpacing`, NOT `letterSpacing` (the latter is silently ignored):

```javascript
s.addText('WIDE LETTERS', { x: 1, y: 1, w: 8, h: 1, charSpacing: 6 });
```

### Mixed Formatting

```javascript
s.addText(
  [
    { text: 'Bold part ', options: { bold: true } },
    { text: 'and italic part', options: { italic: true } },
  ],
  { x: 1, y: 3, w: 8, h: 1 },
);
```

### Line Breaks

Each text item except the last needs `breakLine: true`, or they merge onto one line:

```javascript
s.addText(
  [
    { text: 'First line', options: { breakLine: true } },
    { text: 'Second line', options: { breakLine: true } },
    { text: 'Third line' },
  ],
  { x: 0.5, y: 0.5, w: 8, h: 2 },
);
```

### Alignment with Adjacent Elements

Text boxes have internal padding by default. Set `margin: 0` when text must align with shapes or icons at the same coordinate:

```javascript
s.addText('Flush Left', { x: 0.5, y: 0.3, w: 9, h: 0.6, margin: 0 });
```

---

## Bullet and Numbered Lists

```javascript
s.addText(
  [
    { text: 'Item A', options: { bullet: true, breakLine: true } },
    { text: 'Item B', options: { bullet: true, breakLine: true } },
    { text: 'Item C', options: { bullet: true } },
  ],
  { x: 0.5, y: 0.5, w: 8, h: 3 },
);
```

Nested items: `{ text: "Sub-item", options: { bullet: true, indentLevel: 1 } }`
Numbered: `{ text: "First", options: { bullet: { type: "number" }, breakLine: true } }`

**Never use the `\u2022` character for bullets** — it stacks with the automatic bullet and creates a double-bullet.

---

## Drawing Shapes

```javascript
s.addShape(deck.shapes.RECTANGLE, {
  x: 0.5,
  y: 0.8,
  w: 1.5,
  h: 3,
  fill: { color: 'FF0000' },
  line: { color: '000000', width: 2 },
});

s.addShape(deck.shapes.OVAL, {
  x: 4,
  y: 1,
  w: 2,
  h: 2,
  fill: { color: '0000FF' },
});

s.addShape(deck.shapes.LINE, {
  x: 1,
  y: 3,
  w: 5,
  h: 0,
  line: { color: 'FF0000', width: 3, dashType: 'dash' },
});
```

### Transparency, Corners, Shadows

```javascript
// Semi-transparent fill
s.addShape(deck.shapes.RECTANGLE, {
  x: 1,
  y: 1,
  w: 3,
  h: 2,
  fill: { color: '0088CC', transparency: 50 },
});

// Rounded corners (ROUNDED_RECTANGLE only, not RECTANGLE)
s.addShape(deck.shapes.ROUNDED_RECTANGLE, {
  x: 1,
  y: 1,
  w: 3,
  h: 2,
  fill: { color: 'FFFFFF' },
  rectRadius: 0.1,
});

// Drop shadow
s.addShape(deck.shapes.RECTANGLE, {
  x: 1,
  y: 1,
  w: 3,
  h: 2,
  fill: { color: 'FFFFFF' },
  shadow: { type: 'outer', color: '000000', blur: 6, offset: 2, angle: 135, opacity: 0.15 },
});
```

Shadow properties:

- `type`: `"outer"` or `"inner"`
- `color`: 6-character hex, no `#`
- `blur`: 0-100 pt
- `offset`: 0-200 pt (must be non-negative; negative values corrupt the file)
- `angle`: 0-359 degrees (135 = bottom-right; for upward shadow e.g. footer bars, use 270)
- `opacity`: 0.0-1.0 (never encode transparency in the color string)

Gradient fills have no native API. Use a pre-rendered gradient image as background instead.

---

## Placing Images

```javascript
// From local file
s.addImage({ path: 'images/photo.png', x: 1, y: 1, w: 5, h: 3 });

// From URL
s.addImage({ path: 'https://example.com/img.jpg', x: 1, y: 1, w: 5, h: 3 });

// From base64 (no file I/O, faster)
s.addImage({ data: 'image/png;base64,iVBORw0KGgo...', x: 1, y: 1, w: 5, h: 3 });
```

Sizing modes:

- `{ sizing: { type: "contain", w: 4, h: 3 } }` — fit inside, keep ratio
- `{ sizing: { type: "cover", w: 4, h: 3 } }` — fill area, crop if needed
- `{ sizing: { type: "crop", x: 0.5, y: 0.5, w: 2, h: 2 } }` — cut a region

Aspect-ratio-preserving manual placement:

```javascript
const origW = 1978,
  origH = 923,
  targetH = 3.0;
const scaledW = targetH * (origW / origH);
const cx = (10 - scaledW) / 2;
s.addImage({ path: 'img.png', x: cx, y: 1.2, w: scaledW, h: targetH });
```

Additional properties: `rotate`, `rounding` (circular crop), `transparency`, `altText`, `hyperlink: { url }`.

Supported formats: PNG, JPG, GIF (animated in Microsoft 365), SVG (modern PowerPoint).

---

## Icon Rendering via react-icons

Rasterize SVG icons to PNG so they work universally:

```javascript
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const sharp = require('sharp');
const { FaCheckCircle } = require('react-icons/fa');

function toSvgMarkup(Icon, color = '#000', px = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(Icon, { color, size: String(px) }),
  );
}

async function iconAsBase64(Icon, color, px = 256) {
  const svg = toSvgMarkup(Icon, color, px);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return 'image/png;base64,' + buf.toString('base64');
}

const b64 = await iconAsBase64(FaCheckCircle, '#4472C4');
s.addImage({ data: b64, x: 1, y: 1, w: 0.5, h: 0.5 });
```

Use px >= 256 for sharp results. The pixel value controls rasterization resolution, while `w`/`h` control display size in inches.

Available icon sets: `react-icons/fa` (Font Awesome), `/md` (Material Design), `/hi` (Heroicons), `/bi` (Bootstrap Icons).

---

## Slide Backgrounds

```javascript
s.background = { color: 'F1F1F1' };
s.background = { color: 'FF3399', transparency: 50 };
s.background = { path: 'https://example.com/bg.jpg' };
s.background = { data: 'image/png;base64,iVBORw0KGgo...' };
```

---

## Data Tables

```javascript
s.addTable(
  [
    ['Header 1', 'Header 2'],
    ['Value A', 'Value B'],
  ],
  {
    x: 1,
    y: 1,
    w: 8,
    h: 2,
    border: { pt: 1, color: '999999' },
    fill: { color: 'F1F1F1' },
  },
);
```

Styled cells and column merging:

```javascript
let rows = [
  [
    { text: 'Title', options: { fill: { color: '6699CC' }, color: 'FFFFFF', bold: true } },
    'Detail',
  ],
  [{ text: 'Spanning both', options: { colspan: 2 } }],
];
s.addTable(rows, { x: 1, y: 3.5, w: 8, colW: [4, 4] });
```

---

## Charts

```javascript
// Column
s.addChart(
  deck.charts.BAR,
  [
    {
      name: 'Revenue',
      labels: ['Q1', 'Q2', 'Q3', 'Q4'],
      values: [4500, 5500, 6200, 7100],
    },
  ],
  { x: 0.5, y: 0.6, w: 6, h: 3, barDir: 'col', showTitle: true, title: 'Quarterly Revenue' },
);

// Line
s.addChart(
  deck.charts.LINE,
  [
    {
      name: 'Temp',
      labels: ['Jan', 'Feb', 'Mar'],
      values: [32, 35, 42],
    },
  ],
  { x: 0.5, y: 4, w: 6, h: 3, lineSize: 3, lineSmooth: true },
);

// Pie
s.addChart(
  deck.charts.PIE,
  [
    {
      name: 'Share',
      labels: ['X', 'Y', 'Z'],
      values: [35, 45, 20],
    },
  ],
  { x: 7, y: 1, w: 5, h: 4, showPercent: true },
);
```

### Making Charts Look Modern

Defaults look dated. Apply a clean style:

```javascript
s.addChart(deck.charts.BAR, data, {
  x: 0.5,
  y: 1,
  w: 9,
  h: 4,
  barDir: 'col',
  chartColors: ['0D9488', '14B8A6', '5EEAD4'],
  chartArea: { fill: { color: 'FFFFFF' }, roundedCorners: true },
  catAxisLabelColor: '64748B',
  valAxisLabelColor: '64748B',
  valGridLine: { color: 'E2E8F0', size: 0.5 },
  catGridLine: { style: 'none' },
  showValue: true,
  dataLabelPosition: 'outEnd',
  dataLabelColor: '1E293B',
  showLegend: false,
});
```

Available chart types: BAR, LINE, PIE, DOUGHNUT, SCATTER, BUBBLE, RADAR.

For charts via python-pptx, see [python-charts.md](python-charts.md).

---

## Reusable Slide Masters

```javascript
deck.defineSlideMaster({
  title: 'DARK_OPENER',
  background: { color: '283A5E' },
  objects: [
    {
      placeholder: { options: { name: 'title', type: 'title', x: 1, y: 2, w: 8, h: 2 } },
    },
  ],
});

let opener = deck.addSlide({ masterName: 'DARK_OPENER' });
opener.addText('Welcome', { placeholder: 'title' });
```

---

## Traps to Avoid

These cause corrupt files, visual glitches, or silent failures:

1. **Hex colors must be 6 characters without `#`** — `"FF0000"` works, `"#FF0000"` corrupts the file
2. **Never encode opacity in the color string** — `"00000020"` (8-char hex) corrupts; use `opacity: 0.12` as a separate property
3. **Never use `\u2022` for bullets** — causes doubled bullets; use `bullet: true`
4. **Always set `breakLine: true`** between text array items or they render on the same line
5. **Avoid `lineSpacing` with bulleted lists** — creates excessive gaps; use `paraSpaceAfter` instead
6. **Create a fresh `pptxgen()` per file** — reusing an instance across files produces corrupt output
7. **Never share option objects between calls** — PptxGenJS mutates them in-place (converts values to EMU). Use factory functions:
   ```javascript
   const mkShadow = () => ({ type: "outer", blur: 6, offset: 2, color: "000000", opacity: 0.15 });
   s.addShape(deck.shapes.RECTANGLE, { shadow: mkShadow(), ... });
   s.addShape(deck.shapes.RECTANGLE, { shadow: mkShadow(), ... });
   ```
8. **Don't combine ROUNDED_RECTANGLE with rectangular accent overlays** — the rectangle exposes rounded corners. Use RECTANGLE for shapes that need side accent bars.
