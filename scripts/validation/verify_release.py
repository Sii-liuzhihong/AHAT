"""Check release boundaries and pinned-framework fingerprints without training."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Optional read-only comparison of original framework fingerprints")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    errors = []
    forbidden = {"custom", "checkpoint_manager", "original_trainer", "__pycache__"}
    # Caches are generated locally by Python; they must not enter the distribution.
    for path in root.rglob("*.py"):
        if any(part in {"build", "dist"} or part.endswith(".egg-info") for part in path.relative_to(root).parts):
            continue
        if any(part in forbidden - {"__pycache__"} for part in path.relative_to(root).parts):
            errors.append(f"Excluded component in release: {path.relative_to(root)}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if "robot_planning" not in path.relative_to(root).parts and "tgpo" not in path.relative_to(root).parts:
            continue
        if "tests" in path.relative_to(root).parts or "scripts" in path.relative_to(root).parts:
            continue
        for node in ast.walk(tree):
            imports = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            if any(name.split(".")[0] in {"policy_iteration", "iter", "Call_LLM", "custom"} for name in imports):
                errors.append(f"Private-workspace import: {path.relative_to(root)}:{node.lineno}")
    manifest = json.loads((root / "docs/tgpo/framework_manifest.json").read_text())
    checked = 0
    for row in manifest["files"]:
        current = root / row["path"]
        expected = row.get("public_sha256")
        if expected is not None:
            if not current.is_file() or digest(current) != expected:
                errors.append(f"Released framework fingerprint changed: {row['path']}")
            checked += 1
        if args.source_root and row.get("source_sha256") is not None:
            source = Path(args.source_root) / "RL_framework/verl_fix_cot" / row["path"]
            if not source.is_file() or digest(source) != row["source_sha256"]:
                errors.append(f"Original framework fingerprint changed: {row['path']}")
    print(json.dumps(dict(upstream_commit=manifest["upstream_commit"], framework_files_checked=checked,
                          errors=errors), indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
