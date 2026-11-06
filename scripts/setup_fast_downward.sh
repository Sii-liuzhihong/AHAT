#!/usr/bin/env bash
# Fetch a verified release; keep GPL-licensed solver sources outside our package.
set -euo pipefail

TGPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TGPO_SOLVER_DIR="${1:-${TGPO_ROOT}/third_party/fast-downward}"
TGPO_SOLVER_PYTHON="${PYTHON:-python3}"
TGPO_BUILD_JOBS="${BUILD_JOBS:-4}"

"${TGPO_SOLVER_PYTHON}" - "${TGPO_SOLVER_DIR}" "${TGPO_BUILD_JOBS}" <<'PY'
import hashlib
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

destination = pathlib.Path(sys.argv[1]).expanduser().resolve()
jobs = int(sys.argv[2])
if jobs < 1:
    raise SystemExit('BUILD_JOBS must be positive')
version = 'release-24.06.1'
expected = '994374021146f5d587a3559af016a24d56c404cbbed3d6d431f28c58b5e8b802'
url = f'https://github.com/aibasel/downward/archive/refs/tags/{version}.tar.gz'
if destination.exists():
    raise SystemExit(f'{destination} exists; choose an empty destination or use FAST_DOWNWARD_PATH')
with urllib.request.urlopen(url, timeout=120) as response:
    archive = response.read()
actual = hashlib.sha256(archive).hexdigest()
if actual != expected:
    raise SystemExit(f'Archive checksum mismatch: {actual}')
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='tgpo-downward-') as tmp:
    archive_path = pathlib.Path(tmp) / 'source.tar.gz'
    archive_path.write_bytes(archive)
    with tarfile.open(archive_path) as source:
        # Verify paths even though the downloaded bytes have a pinned digest.
        for member in source.getmembers():
            resolved = (pathlib.Path(tmp) / member.name).resolve()
            if not resolved.is_relative_to(pathlib.Path(tmp).resolve()):
                raise SystemExit('Unsafe archive path')
        source.extractall(tmp)
    (pathlib.Path(tmp) / f'downward-{version}').rename(destination)
subprocess.run([sys.executable, 'build.py', 'release', f'-j{jobs}'], cwd=destination, check=True)
print(f'Fast Downward built at {destination}')
print(f'export FAST_DOWNWARD_PATH={destination / "fast-downward.py"}')
PY
