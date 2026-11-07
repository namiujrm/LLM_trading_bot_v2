"""Stop Hook Handler: Prevents ending turn if verification gates fail."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    raw_stdin = sys.stdin.read()
    _payload: object = json.loads(raw_stdin) if raw_stdin.strip() else {}
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"

    if not src_dir.exists() or not any(src_dir.rglob("*.py")):
        print(json.dumps({"decision": "allow"}))
        return

    commands: list[list[str]] = [
        ["uv", "run", "ruff", "check", "src", "tests", "scripts"],
        ["uv", "run", "pyright", "src", "tests", "scripts"],
        ["uv", "run", "python", "scripts/check_no_defaults_or_hardcode.py"],
        ["uv", "run", "lint-imports"],
        ["uv", "run", "pytest", "tests", "-q"],
    ]

    for cmd in commands:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if proc.returncode != 0:
            err_snippet = (proc.stdout + "\n" + proc.stderr).strip()[-1200:]
            print(
                json.dumps(
                    {
                        "decision": "continue",
                        "reason": (
                            f"[Verify-Loop Gate Blocked Stop] Command `{' '.join(cmd)}` failed.\n"
                            f"Fix the following error before ending the turn:\n{err_snippet}"
                        ),
                    }
                )
            )
            return

    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
