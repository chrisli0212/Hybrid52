"""Text and metadata extraction from PPTX files."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation


def extract_slides(pptx_path: str, notes: bool = True) -> list[dict]:
    prs = Presentation(pptx_path)
    result = []

    for idx, slide in enumerate(prs.slides, 1):
        record = {"number": idx, "heading": "", "content": [], "notes": ""}

        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            raw_parts = []
            for p in shape.text_frame.paragraphs:
                ptext = p.text.replace("\x0b", "\n")
                if not ptext.strip():
                    continue
                raw_parts.append(ptext)
            block = "\n".join(raw_parts)
            if not block:
                continue
            if shape == slide.shapes.title:
                record["heading"] = block
            else:
                record["content"].append(block)

        if notes and slide.has_notes_slide:
            tf = slide.notes_slide.notes_text_frame
            if tf:
                record["notes"] = tf.text

        result.append(record)

    return result


def format_plain(slides: list[dict]) -> str:
    sections = []
    for s in slides:
        header = f"=== Slide {s['number']} ==="
        parts = [header]
        if s["heading"]:
            for line in s["heading"].split("\n"):
                parts.append(f"  {line}")
        for block in s["content"]:
            for line in block.split("\n"):
                parts.append(f"  {line}")
        if s["notes"]:
            parts.append(f"  [Speaker Notes] {s['notes']}")
        sections.append("\n".join(parts))
    return "\n\n".join(sections)


def format_json(slides: list[dict]) -> str:
    return json.dumps(slides, indent=2, ensure_ascii=False)
