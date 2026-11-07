"""Architecture compliance and Invariant #9 tests."""

from __future__ import annotations

from pathlib import Path

from scripts.check_no_defaults_or_hardcode import scan_directory


def test_zero_defaults_and_zero_hardcode_in_src() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src"
    violations = scan_directory(src_dir, repo_root)
    assert violations == [], f"Found Invariant #9 violations in src/: {violations}"
