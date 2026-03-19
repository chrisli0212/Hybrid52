#!/usr/bin/env python3
"""Unified CLI for presentation file operations.

Usage:
    python deck.py check-env
    python deck.py read      <pptx> [--format json] [--skip-notes] [--out FILE]
    python deck.py thumbnails <pptx> [--columns N]
    python deck.py unpack    <pptx> <workspace/>
    python deck.py pack      <workspace/> <output.pptx> [--original FILE]
    python deck.py render    <pptx> [--range PAGES] [--dpi N] [--dest DIR]
    python deck.py clone     <workspace/> <source.xml>
    python deck.py purge     <workspace/>
    python deck.py create    <spec.json> <output.pptx>
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="deck.py", description="Presentation toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_read = sub.add_parser("read", help="Extract text content")
    p_read.add_argument("pptx")
    p_read.add_argument("--format", choices=["text", "json"], default="text")
    p_read.add_argument("--skip-notes", action="store_true")
    p_read.add_argument("--out", help="Write to file instead of stdout")

    p_thumb = sub.add_parser("thumbnails", help="Generate slide thumbnail grid")
    p_thumb.add_argument("pptx")
    p_thumb.add_argument("--columns", type=int, default=3)

    p_unpack = sub.add_parser("unpack", help="Extract and format PPTX for editing")
    p_unpack.add_argument("pptx")
    p_unpack.add_argument("workspace")

    p_pack = sub.add_parser("pack", help="Reassemble workspace into PPTX")
    p_pack.add_argument("workspace")
    p_pack.add_argument("output")
    p_pack.add_argument("--original", help="Source file for validation")
    p_pack.add_argument("--skip-checks", action="store_true")

    p_render = sub.add_parser("render", help="Render slides to images")
    p_render.add_argument("pptx")
    p_render.add_argument("--range", help="Page range e.g. '3-5'")
    p_render.add_argument("--dpi", type=int, default=150)
    p_render.add_argument("--dest", default=".")

    p_clone = sub.add_parser("clone", help="Clone slide or create from layout")
    p_clone.add_argument("workspace")
    p_clone.add_argument("source", help="slide2.xml or slideLayout3.xml")

    p_purge = sub.add_parser("purge", help="Remove orphaned files from workspace")
    p_purge.add_argument("workspace")

    p_create = sub.add_parser("create", help="Create deck from JSON spec")
    p_create.add_argument("spec", help="JSON spec file path")
    p_create.add_argument("output", help="Output PPTX path")

    p_check = sub.add_parser("check-env", help="Check environment dependencies")
    p_check.add_argument("--install", action="store_true", help="Auto-install missing deps")

    args = parser.parse_args()

    dispatch = {
        "read": _cmd_read,
        "thumbnails": _cmd_thumbnails,
        "unpack": _cmd_unpack,
        "pack": _cmd_pack,
        "render": _cmd_render,
        "clone": _cmd_clone,
        "purge": _cmd_purge,
        "create": _cmd_create,
        "check-env": _cmd_check_env,
    }
    try:
        dispatch[args.command](args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_read(args):
    from internals.content import extract_slides, format_plain, format_json

    slides = extract_slides(args.pptx, notes=not args.skip_notes)
    text = format_json(slides) if args.format == "json" else format_plain(slides)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Written to {args.out}")
    else:
        print(text)


def _cmd_thumbnails(args):
    from internals.gridshot import generate_grid
    outputs = generate_grid(args.pptx, columns=args.columns)
    for f in outputs:
        print(f"Created: {f}")


def _cmd_unpack(args):
    from internals.packaging import disassemble
    msg = disassemble(args.pptx, args.workspace)
    print(msg)


def _cmd_pack(args):
    from internals.packaging import reassemble
    msg = reassemble(
        args.workspace, args.output,
        reference=args.original,
        validate=not args.skip_checks,
    )
    print(msg)


def _cmd_render(args):
    from internals.imaging import render_slides
    images = render_slides(args.pptx, page_range=args.range, dpi=args.dpi, dest=args.dest)
    print(f"Rendered {len(images)} slide(s):")
    for img in images:
        print(f"  {img}")


def _cmd_clone(args):
    from internals.structure import clone_or_instantiate
    clone_or_instantiate(args.workspace, args.source)


def _cmd_purge(args):
    from internals.cleanup import sweep_workspace
    removed = sweep_workspace(args.workspace)
    if removed:
        print(f"Removed {len(removed)} orphaned item(s):")
        for r in removed:
            print(f"  {r}")
    else:
        print("No orphans found")


def _cmd_create(args):
    from internals.create import create_deck
    msg = create_deck(args.spec, args.output)
    print(msg)


def _cmd_check_env(args):
    from pathlib import Path
    import importlib.util
    check_env_path = Path(__file__).parent / "check_env.py"
    spec = importlib.util.spec_from_file_location("check_env", check_env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main(install=getattr(args, 'install', False)))


if __name__ == "__main__":
    main()
