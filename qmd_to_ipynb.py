#!/usr/bin/env python3
"""
qmd_to_ipynb.py

Usage:
    python qmd_to_ipynb.py [path/to/file.qmd]

    Without arguments, converts all .qmd files in the current directory.

Behavior:
- Converts a Quarto .qmd to a Jupyter .ipynb in OUTPUT_DIR
- Executable code blocks -> code cells
- All code cells are cleared EXCEPT those marked to keep via:
    #| tags: [noclear]   OR   #| keep: true
- Blocks marked with #| tags: [uncomment] are kept AND have all lines
  uncommented (leading '# ' removed). This implicitly acts as [noclear].
- Blocks marked with #| tags: [remove] are completely removed from the
  output (no cell created), including any trailing blank lines.
- In kept code cells, ALL lines starting with '#|' are removed
- Markdown text is preserved as markdown cells, with each paragraph
  (text separated by blank lines) becoming a separate cell
- Lines starting with '**Answer**:' have everything after '**Answer**:' removed
"""

import sys
import os
import json
import re

# ===== Hard-coded output directory =====
OUTPUT_DIR = "nb"

# --- Helpers -----------------------------------------------------------------

_LANGS_COMMON = {"python", "r", "bash", "sh", "zsh", "julia", "javascript", "js", "typescript", "ts"}

def _is_executable_chunk(langspec: str) -> bool:
    langspec = langspec.strip().lower()
    if langspec.startswith("{") and langspec.endswith("}"):
        return True
    bare = langspec.strip("` ").strip()
    return bare in _LANGS_COMMON

_KEEP_TAG_RE = re.compile(r"^\s*#\|\s*tags\s*:\s*\[([^\]]*)\]", re.IGNORECASE | re.MULTILINE)
_KEEP_FLAG_RE = re.compile(r"^\s*#\|\s*keep\s*:\s*(true|yes|1)\s*$", re.IGNORECASE | re.MULTILINE)

def _get_tags(body: str) -> set:
    """Extract tags from a code block body."""
    m = _KEEP_TAG_RE.search(body)
    if m:
        tags_str = m.group(1)
        return {t.strip().strip("'\"").lower() for t in tags_str.split(",")}
    return set()

def _has_keep_signal(body: str) -> bool:
    if _KEEP_FLAG_RE.search(body):
        return True
    tags = _get_tags(body)
    return bool(tags & {"noclear", "keep", "uncomment"})

def _has_uncomment_signal(body: str) -> bool:
    return "uncomment" in _get_tags(body)

def _has_remove_signal(body: str) -> bool:
    return "remove" in _get_tags(body)

def _uncomment_lines(lines):
    """Remove leading '# ' from lines (single comment prefix)."""
    result = []
    for ln in lines:
        # Remove one '# ' prefix if present (preserving the rest)
        if ln.startswith("# "):
            result.append(ln[2:])
        elif ln.startswith("#") and (len(ln) == 1 or ln[1:].startswith("\n")):
            # Handle bare '#' or '#\n'
            result.append(ln[1:])
        else:
            result.append(ln)
    return result

def _strip_quarto_option_lines(lines):
    """Remove all Quarto option lines that start with '#|' (any indentation)."""
    return [ln for ln in lines if not ln.lstrip().startswith("#|")]

def _strip_answer_content(lines):
    """For lines starting with '**Answer**:', keep only '**Answer**:' and remove the rest."""
    result = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("**Answer**:"):
            # Find the position of '**Answer**:' in the original line (preserving leading whitespace)
            idx = ln.find("**Answer**:")
            # Keep leading whitespace + '**Answer**:' + newline if original had one
            if ln.endswith("\n"):
                result.append(ln[:idx] + "**Answer**:\n")
            else:
                result.append(ln[:idx] + "**Answer**:")
        else:
            result.append(ln)
    return result

def _split_into_paragraphs(text: str) -> list:
    """Split text into paragraphs (separated by one or more blank lines).

    Returns a list of non-empty paragraph strings.
    """
    # Split on one or more blank lines (two or more consecutive newlines)
    # This regex matches: newline + optional whitespace-only lines + newline
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out empty paragraphs and strip each one
    return [p.strip() for p in paragraphs if p.strip()]

# --- Core --------------------------------------------------------------------

FENCE_RE = re.compile(
    r"```([^\n]*)\n"      # opening fence + langspec (group 1)
    r"(.*?)"              # body (group 2), non-greedy
    r"\n```",             # closing fence
    re.DOTALL
)

def qmd_to_ipynb(filepath: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    cells = []
    pos = 0

    for m in FENCE_RE.finditer(content):
        start, end = m.span()
        langspec = m.group(1).strip()
        body = m.group(2)

        # Preceding text -> markdown (split into separate cells per paragraph)
        if start > pos:
            text_chunk = content[pos:start]
            for para in _split_into_paragraphs(text_chunk):
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},  # no tags/metadata carried over
                    "source": _strip_answer_content(para.splitlines(keepends=True))
                })

        if _is_executable_chunk(langspec):
            # Check if block should be removed entirely
            if _has_remove_signal(body):
                # Skip trailing blank lines after the removed block
                trailing = content[end:]
                stripped = trailing.lstrip('\n')
                pos = end + (len(trailing) - len(stripped))
                continue

            keep_code = _has_keep_signal(body)
            if keep_code:
                raw_lines = body.splitlines(keepends=True)
                code_source = _strip_quarto_option_lines(raw_lines)
                if _has_uncomment_signal(body):
                    code_source = _uncomment_lines(code_source)
            else:
                code_source = []
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},  # keep empty to avoid ipynb tags
                "outputs": [],
                "source": code_source
            })
        else:
            # Non-exec fence -> markdown (as literal fenced block)
            fence_text = content[start:end]
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": _strip_answer_content(fence_text.splitlines(keepends=True))
            })

        pos = end

    # Trailing text (split into separate cells per paragraph)
    if pos < len(content):
        tail = content[pos:]
        for para in _split_into_paragraphs(tail):
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": _strip_answer_content(para.splitlines(keepends=True))
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(OUTPUT_DIR, base + ".ipynb")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook saved to {out_path}")

# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: convert all .qmd files in current directory
        import glob
        qmd_files = sorted(glob.glob("*.qmd"))
        if not qmd_files:
            print("No .qmd files found in current directory")
            sys.exit(1)
        for qmd_file in qmd_files:
            qmd_to_ipynb(qmd_file)
    elif len(sys.argv) == 2:
        qmd_to_ipynb(sys.argv[1])
    else:
        print("Usage: python qmd_to_ipynb.py [file.qmd]")
        print("       Without arguments, converts all .qmd files in current directory")
        sys.exit(1)
