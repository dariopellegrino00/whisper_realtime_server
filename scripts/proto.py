from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PROTO_FILE = ROOT / "proto" / "speech.proto"
GEN_DIR = ROOT / "src" / "generated"
FIX_IMPORTS_SCRIPT = ROOT / "fix_proto_imports.py"


def generate() -> int:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "__init__.py").touch(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_FILE.parent}",
        f"--python_out={GEN_DIR}",
        f"--grpc_python_out={GEN_DIR}",
        str(PROTO_FILE),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(FIX_IMPORTS_SCRIPT)], check=True, cwd=ROOT)
    return 0


def clean() -> int:
    for path in GEN_DIR.glob("*.py"):
        path.unlink()
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"generate", "clean"}:
        print("Usage: python scripts/proto.py [generate|clean]")
        return 1

    if sys.argv[1] == "generate":
        return generate()
    return clean()


if __name__ == "__main__":
    raise SystemExit(main())
