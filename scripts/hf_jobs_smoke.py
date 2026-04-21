# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=9.0",
#   "rfdetr>=1.4.1",
#   "matplotlib>=3.9",
# ]
# ///
"""HF Jobs smoke test — proves the LibreYOLO fork installs cleanly on HF compute
and the VisDrone fine-tune pipeline's smoke tests pass end-to-end.

Clones the fork, installs in editable mode (so tests are available), and runs
pytest on tests/smoke/. Exits with pytest's return code.

Expected wall-clock: ~3 minutes on cpu-basic. Cost: well under $0.01.

Local dry-run:
    python scripts/hf_jobs_smoke.py

Submit to HF Jobs:
    hf jobs uv run scripts/hf_jobs_smoke.py --flavor cpu-basic --secrets HF_TOKEN
"""
import os
import subprocess
import sys
from pathlib import Path


FORK_URL = "https://github.com/aalvsz/libreyolo.git"
BRANCH = os.environ.get("LIBREYOLO_BRANCH", "agentic/c-visdrone-finetune")
WORKDIR = Path(os.environ.get("LIBREYOLO_WORKDIR", "/tmp/libreyolo_hf_smoke"))


def sh(cmd: list[str], cwd: Path | None = None) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd)


def main() -> int:
    print("=" * 60, flush=True)
    print(f"LibreYOLO HF Jobs smoke test", flush=True)
    print(f"  fork:   {FORK_URL}", flush=True)
    print(f"  branch: {BRANCH}", flush=True)
    print(f"  work:   {WORKDIR}", flush=True)
    print("=" * 60, flush=True)

    if WORKDIR.exists():
        import shutil
        shutil.rmtree(WORKDIR)

    rc = sh(["git", "clone", "--depth", "1", "--branch", BRANCH, FORK_URL, str(WORKDIR)])
    if rc:
        return rc

    rc = sh([sys.executable, "-m", "pip", "install", "-e", "."], cwd=WORKDIR)
    if rc:
        return rc

    print("-" * 60, flush=True)
    sh([sys.executable, "-c",
        "import torch, libreyolo; print('torch:', torch.__version__); "
        "print('libreyolo:', libreyolo.__version__)"])
    print("-" * 60, flush=True)

    smoke_test = WORKDIR / "tests" / "smoke" / "test_visdrone_finetune_smoke.py"
    print(f"Running smoke tests: {smoke_test}", flush=True)

    rc = sh([
        sys.executable, "-m", "pytest",
        str(smoke_test),
        "-v",
        "-o", "addopts=",   # bypass repo's default -m unit filter
        "--tb=short",
    ], cwd=WORKDIR)
    print(f"pytest exited with {rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
