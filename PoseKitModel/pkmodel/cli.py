import os
import runpy
import sys
from pathlib import Path
from typing import List, Optional

COMMANDS = {
    "train": ("train.py", "Train PoseKitModel"),
    "evaluate": ("evaluate.py", "Evaluate PoseKitModel"),
    "generate-coreml": ("generate_coreml.py", "Convert checkpoint to Core ML"),
    "data-tools": ("data_tools.py", "Dataset utilities"),
    "prepare-coco": ("prepare_coco.py", "Prepare COCO labels"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _print_help() -> None:
    print("Usage: pkmodel <command> [args...]")
    print("")
    print("Commands:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:15} {desc}")
    print("")
    print("Run `pkmodel <command> --help` to see command-specific options.")


def _run_script(script: str, argv: List[str]) -> int:
    repo_root = _repo_root()
    script_path = repo_root / script
    if not script_path.exists():
        print(f"[pkmodel] Script not found: {script_path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    prev_argv = sys.argv
    try:
        sys.argv = [str(script_path)] + argv
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = prev_argv
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    if argv[0] in {"--list", "list"}:
        for name in COMMANDS:
            print(name)
        return 0

    command = argv[0]
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"[pkmodel] Unknown command: {command}", file=sys.stderr)
        _print_help()
        return 2

    script, _ = entry
    return _run_script(script, argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
