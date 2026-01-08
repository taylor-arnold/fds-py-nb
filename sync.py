#!/usr/bin/env python3
"""
sync.py

Syncs files from ../fds-py to this directory:
- data/ (recursively)
- media/ (recursively)
- funs.py

When files differ or are missing, copies FROM ../fds-py TO here.
"""

import os
import shutil
import filecmp
from pathlib import Path

SOURCE_DIR = Path("../fds-py")
DEST_DIR = Path(".")

SYNC_DIRS = ["data", "media"]
SYNC_FILES = ["funs.py"]


def sync_file(src: Path, dst: Path) -> bool:
    """Sync a single file. Returns True if file was copied."""
    if not src.exists():
        print(f"  WARNING: Source missing: {src}")
        return False

    # Check if destination exists and is identical
    if dst.exists():
        if filecmp.cmp(src, dst, shallow=False):
            return False  # Files are identical
        else:
            print(f"  Updated: {dst}")
    else:
        # Ensure parent directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Added:   {dst}")

    shutil.copy2(src, dst)
    return True


def sync_directory(dirname: str) -> tuple[int, int]:
    """Sync a directory recursively. Returns (checked_count, copied_count)."""
    src_dir = SOURCE_DIR / dirname
    dst_dir = DEST_DIR / dirname

    if not src_dir.exists():
        print(f"  WARNING: Source directory missing: {src_dir}")
        return 0, 0

    checked = 0
    copied = 0

    for src_path in src_dir.rglob("*"):
        if src_path.is_file():
            rel_path = src_path.relative_to(src_dir)
            dst_path = dst_dir / rel_path
            checked += 1
            if sync_file(src_path, dst_path):
                copied += 1

    return checked, copied


def main():
    print(f"Syncing from {SOURCE_DIR.resolve()} to {DEST_DIR.resolve()}\n")

    total_checked = 0
    total_copied = 0

    # Sync directories
    for dirname in SYNC_DIRS:
        print(f"Syncing {dirname}/...")
        checked, copied = sync_directory(dirname)
        total_checked += checked
        total_copied += copied
        if copied == 0:
            print(f"  (no changes)")

    # Sync individual files
    print("Syncing individual files...")
    for filename in SYNC_FILES:
        src = SOURCE_DIR / filename
        dst = DEST_DIR / filename
        total_checked += 1
        if sync_file(src, dst):
            total_copied += 1

    if total_copied == 0 and total_checked > 0:
        print(f"  (no changes)")

    # Summary
    print(f"\nSummary: {total_checked} files checked, {total_copied} files copied")


if __name__ == "__main__":
    main()
