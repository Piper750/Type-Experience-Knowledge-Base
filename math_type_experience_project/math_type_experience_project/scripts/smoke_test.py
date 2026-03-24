from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    python_exec = sys.executable
    build_cmd = [python_exec, str(PROJECT_ROOT / "scripts" / "build_kb.py"), "--config", args.config]
    eval_cmd = [python_exec, str(PROJECT_ROOT / "scripts" / "evaluate.py"), "--config", args.config]
    ablation_cmd = [python_exec, str(PROJECT_ROOT / "scripts" / "run_ablation.py"), "--config", args.config]

    for cmd in [build_cmd, eval_cmd, ablation_cmd]:
        print(f"[smoke_test] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    print("[smoke_test] all steps finished successfully.")


if __name__ == "__main__":
    main()
