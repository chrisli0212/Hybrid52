"""Create a new presentation from scratch using python-pptx."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


def create_deck(spec_path: str, output_path: str):
    """Create a deck from a JSON spec file.

    Spec format:
    {
        "width": 13.333,   // slide width in inches (default 13.333 for 16:9)
        "height": 7.5,     // slide height in inches (default 7.5)
        "slides": [
            {
                "background": "1E2761",          // optional hex color
                "elements": [
                    {
                        "type": "text",
                        "text": "Hello World",
                        "x": 1, "y": 1, "w": 8, "h": 2,
                        "font_size": 36,
                        "color": "FFFFFF",
                        "bold": true,
                        "align": "center",       // left|center|right
                        "font": "Arial"
                    },
                    {
                        "type": "image",
                        "path": "chart.png",
                        "x": 1, "y": 2, "w": 8, "h": 4
                    },
                    {
                        "type": "shape",
                        "shape": "rectangle",    // rectangle|rounded_rectangle|oval
                        "x": 0, "y": 0, "w": 13.333, "h": 0.8,
                        "fill": "D4A017"
                    }
                ]
            }
        ]
    }
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    prs = Presentation()

    prs.slide_width = Inches(spec.get("width", 13.333))
    prs.slide_height = Inches(spec.get("height", 7.5))

    blank_layout = prs.slide_layouts[6]

    align_map = {
        "left": PP_ALIGN.LEFT,
        "center": PP_ALIGN.CENTER,
        "right": PP_ALIGN.RIGHT,
    }

    for slide_spec in spec.get("slides", []):
        slide = prs.slides.add_slide(blank_layout)

        bg_color = slide_spec.get("background")
        if bg_color:
            bg = slide.background
            fill = bg.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor.from_string(bg_color)

        for elem in slide_spec.get("elements", []):
            etype = elem["type"]

            if etype == "text":
                txBox = slide.shapes.add_textbox(
                    Inches(elem["x"]), Inches(elem["y"]),
                    Inches(elem["w"]), Inches(elem["h"]),
                )
                tf = txBox.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = elem["text"]
                p.font.size = Pt(elem.get("font_size", 18))
                color = elem.get("color", "000000")
                p.font.color.rgb = RGBColor.from_string(color)
                p.font.bold = elem.get("bold", False)
                p.font.name = elem.get("font", "Arial")
                p.alignment = align_map.get(elem.get("align", "left"), PP_ALIGN.LEFT)

            elif etype == "image":
                slide.shapes.add_picture(
                    elem["path"],
                    Inches(elem["x"]), Inches(elem["y"]),
                    Inches(elem["w"]), Inches(elem["h"]),
                )

            elif etype == "shape":
                from pptx.enum.shapes import MSO_SHAPE
                shape_map = {
                    "rectangle": MSO_SHAPE.RECTANGLE,
                    "rounded_rectangle": MSO_SHAPE.ROUNDED_RECTANGLE,
                    "oval": MSO_SHAPE.OVAL,
                }
                shape_type = shape_map.get(elem.get("shape", "rectangle"), MSO_SHAPE.RECTANGLE)
                shp = slide.shapes.add_shape(
                    shape_type,
                    Inches(elem["x"]), Inches(elem["y"]),
                    Inches(elem["w"]), Inches(elem["h"]),
                )
                fill_color = elem.get("fill")
                if fill_color:
                    shp.fill.solid()
                    shp.fill.fore_color.rgb = RGBColor.from_string(fill_color)
                line_color = elem.get("line")
                if line_color:
                    shp.line.color.rgb = RGBColor.from_string(line_color)

    prs.save(output_path)
    return f"Created {output_path} with {len(spec.get('slides', []))} slide(s)"
