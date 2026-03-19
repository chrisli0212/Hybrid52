#!/usr/bin/env python3
"""Unified command-line interface for Word document operations.

Subcommands:
    extract          Unpack a .docx into editable XML files
    assemble         Repackage edited XML back into a .docx
    verify           Validate document structure and content
    render           Convert pages to images for visual inspection
    convert          Transform between document formats
    accept-changes   Produce a clean copy with all revisions accepted
    annotate         Insert comments into an unpacked document
"""

import argparse
import sys
from pathlib import Path


def cmd_extract(args):
    from internals.packaging import decompose_archive
    _, msg = decompose_archive(
        args.file,
        args.output_dir,
        consolidate_runs=args.merge_runs,
        consolidate_revisions=args.consolidate_revisions,
    )
    print(msg)
    if msg.startswith("Error"):
        sys.exit(1)


def cmd_assemble(args):
    from internals.packaging import compose_archive
    _, msg = compose_archive(
        args.workspace,
        args.output,
        reference_file=args.original,
        run_verification=not args.skip_verify,
    )
    print(msg)
    if msg.startswith("Error"):
        sys.exit(1)


def cmd_verify(args):
    from internals.integrity.orchestrator import run_checks
    success = run_checks(
        args.path,
        original=args.original,
        auto_fix=args.auto_fix,
        author=args.author,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


def cmd_render(args):
    from internals.runtime.renderer import pages_to_images
    pages_to_images(
        args.file,
        page_range=args.range,
        resolution=args.dpi,
        output_dir=args.dest,
    )


def cmd_convert(args):
    from internals.runtime.converter import transform_format
    transform_format(args.file, target_format=args.to)


def cmd_accept_changes(args):
    from internals.runtime.revision_acceptor import finalize_revisions
    _, msg = finalize_revisions(args.input, args.output)
    print(msg)
    if msg.startswith("Error"):
        sys.exit(1)


def cmd_annotate(args):
    from internals.content import insert_annotation
    para_id, msg = insert_annotation(
        args.workspace,
        args.id,
        args.text,
        author=args.author,
        initials=args.initials,
        parent_id=args.reply_to,
    )
    print(msg)
    if msg.startswith("Error"):
        sys.exit(1)


def build_parser():
    root = argparse.ArgumentParser(
        prog="docx_tool",
        description="Word document toolkit — read, create, edit, validate, render",
    )
    sub = root.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Unpack .docx into editable XML")
    p_extract.add_argument("file", help="Source .docx file")
    p_extract.add_argument("output_dir", help="Destination directory")
    p_extract.add_argument("--merge-runs", action="store_true", default=True,
                           help="Merge adjacent runs with identical formatting")
    p_extract.add_argument("--no-merge-runs", dest="merge_runs", action="store_false")
    p_extract.add_argument("--consolidate-revisions", action="store_true", default=True,
                           help="Merge adjacent tracked changes from same author")
    p_extract.add_argument("--no-consolidate-revisions", dest="consolidate_revisions",
                           action="store_false")
    p_extract.set_defaults(func=cmd_extract)

    p_assemble = sub.add_parser("assemble", help="Repackage XML into .docx")
    p_assemble.add_argument("workspace", help="Unpacked document directory")
    p_assemble.add_argument("output", help="Output .docx file path")
    p_assemble.add_argument("--original", help="Original file for validation baseline")
    p_assemble.add_argument("--skip-verify", action="store_true",
                            help="Skip validation step")
    p_assemble.set_defaults(func=cmd_assemble)

    p_verify = sub.add_parser("verify", help="Validate document structure")
    p_verify.add_argument("path", help="Unpacked directory or .docx file")
    p_verify.add_argument("--original", help="Original file for comparison")
    p_verify.add_argument("--auto-fix", action="store_true",
                          help="Attempt automatic repair of common issues")
    p_verify.add_argument("--author", default="Verdent",
                          help="Author name for revision validation")
    p_verify.add_argument("-v", "--verbose", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_render = sub.add_parser("render", help="Convert pages to images")
    p_render.add_argument("file", help=".docx file to render")
    p_render.add_argument("--range", help="Page range (e.g. 1-3)")
    p_render.add_argument("--dpi", type=int, default=150, help="Resolution (default 150)")
    p_render.add_argument("--dest", help="Output directory for images")
    p_render.set_defaults(func=cmd_render)

    p_convert = sub.add_parser("convert", help="Transform document format")
    p_convert.add_argument("file", help="Source file")
    p_convert.add_argument("--to", required=True, help="Target format (docx, pdf)")
    p_convert.set_defaults(func=cmd_convert)

    p_accept = sub.add_parser("accept-changes", help="Accept all tracked changes")
    p_accept.add_argument("input", help="Input .docx with tracked changes")
    p_accept.add_argument("output", help="Output .docx (clean)")
    p_accept.set_defaults(func=cmd_accept_changes)

    p_annotate = sub.add_parser("annotate", help="Insert comments")
    p_annotate.add_argument("workspace", help="Unpacked document directory")
    p_annotate.add_argument("id", type=int, help="Comment ID (must be unique)")
    p_annotate.add_argument("text", help="Comment text (pre-escaped XML)")
    p_annotate.add_argument("--author", default="Verdent", help="Author name")
    p_annotate.add_argument("--initials", default="V", help="Author initials")
    p_annotate.add_argument("--reply-to", type=int, help="Parent comment ID for replies")
    p_annotate.set_defaults(func=cmd_annotate)

    return root


if __name__ == "__main__":
    parser = build_parser()
    parsed = parser.parse_args()
    parsed.func(parsed)
