from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"
REQUIREMENTS = ROOT / "requirements.txt"


def _run(*args: str) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def _export_requirements(output_file: Path) -> None:
    _run(
        "uv",
        "export",
        "--frozen",
        "--no-header",
        "--no-emit-project",
        "--format",
        "requirements.txt",
        "--output-file",
        str(output_file),
    )


def sync() -> int:
    _run("uv", "lock")
    _export_requirements(REQUIREMENTS)
    return 0


def check() -> int:
    if not PYPROJECT.exists():
        print("pyproject.toml not found", file=sys.stderr)
        return 1
    if not LOCKFILE.exists():
        print("uv.lock not found; run `python scripts/deps.py sync` first", file=sys.stderr)
        return 1
    if not REQUIREMENTS.exists():
        print("requirements.txt not found; run `python scripts/deps.py sync` first", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        expected = Path(tmpdir) / "requirements.txt"
        _export_requirements(expected)
        if expected.read_text() != REQUIREMENTS.read_text():
            print(
                "requirements.txt is out of sync with pyproject.toml/uv.lock. "
                "Run `python scripts/deps.py sync`.",
                file=sys.stderr,
            )
            return 1

    return 0


def main() -> int:
    if shutil.which("uv") is None:
        print("uv is required to manage dependencies", file=sys.stderr)
        return 1

    if len(sys.argv) != 2 or sys.argv[1] not in {"sync", "check"}:
        print("Usage: python scripts/deps.py [sync|check]", file=sys.stderr)
        return 1

    if sys.argv[1] == "sync":
        return sync()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
