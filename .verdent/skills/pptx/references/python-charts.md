# python-pptx Chart Recipes

## Column Chart

```python
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])

data = CategoryChartData()
data.categories = ["Q1", "Q2", "Q3", "Q4"]
data.add_series("Series A", (19.2, 21.4, 16.7, 28.3))
data.add_series("Series B", (22.5, 28.1, 25.9, 31.2))

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(1), Inches(2), Inches(8), Inches(4.5), data
).chart

prs.save("result.pptx")
```

## Pie Chart

```python
data = CategoryChartData()
data.categories = ["Apple", "Banana", "Cherry", "Date"]
data.add_series("Share", (0.35, 0.25, 0.22, 0.18))

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.PIE, left, top, width, height, data
).chart
```

## Line Chart

```python
data = CategoryChartData()
data.categories = ["Jan", "Feb", "Mar", "Apr", "May"]
data.add_series("Revenue", (4.5, 5.2, 4.8, 6.1, 7.3))
data.add_series("Expenses", (3.2, 3.5, 3.8, 4.1, 4.0))

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.LINE_MARKERS, left, top, width, height, data
).chart
chart.series[0].smooth = True
```

## XY Scatter

```python
from pptx.chart.data import XyChartData

data = XyChartData()
series = data.add_series("Points")
for x, y in [(1.2, 2.5), (2.3, 4.1), (3.5, 3.8)]:
    series.add_data_point(x, y)

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.XY_SCATTER, left, top, width, height, data
).chart
```

## Styling

```python
from pptx.enum.chart import XL_LEGEND_POSITION

chart.has_title = True
chart.chart_title.text_frame.text = "Report Title"

chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.BOTTOM
```

## Available Types

| Type             | Constant           |
| ---------------- | ------------------ |
| Clustered Column | `COLUMN_CLUSTERED` |
| Stacked Column   | `COLUMN_STACKED`   |
| Clustered Bar    | `BAR_CLUSTERED`    |
| Line             | `LINE`             |
| Line + Markers   | `LINE_MARKERS`     |
| Pie              | `PIE`              |
| Area             | `AREA`             |
| XY Scatter       | `XY_SCATTER`       |
