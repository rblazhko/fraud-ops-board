#!/usr/bin/env python3
"""Optional RNG validation.

Default: prove seed reproducibility for this project's Generator wiring.
If rng-diagnostics is installed (https://github.com/rblazhko/rng-diagnostics),
we try a light CLI call; otherwise we print skip notes and exit 0.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def load_seed(path: Path) -> int:
    with path.open() as f:
        return int(yaml.safe_load(f)["master_seed"])


def reproducibility_check(seed: int) -> None:
    def draw(s: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(s))
        return rng.random(64)

    a = draw(seed)
    b = draw(seed)
    if not np.allclose(a, b):
        raise SystemExit("PCG64 stream not reproducible for master_seed")
    c = draw(seed + 1)
    if np.allclose(a, c):
        raise SystemExit("different seeds unexpectedly matched")
    print(f"ok: PCG64(master_seed={seed}) reproducible ({len(a)} draws)")


def try_rng_diagnostics(seed: int) -> None:
    exe = shutil.which("rng-diagnostics")
    if exe is None:
        try:
            import importlib.util

            if importlib.util.find_spec("rng_diagnostics") is None:
                print(
                    "skip: rng-diagnostics not installed "
                    "(see https://github.com/rblazhko/rng-diagnostics)"
                )
                return
            cmd = [sys.executable, "-m", "rng_diagnostics", "--help"]
        except Exception:
            print("skip: rng-diagnostics not installed")
            return
    else:
        cmd = [exe, "--help"]

    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=15)
        print(f"note: rng-diagnostics available via `{cmd[0]}` — run its suite separately")
        print(f"      example seed to plug in: {seed}")
    except Exception as exc:
        print(f"skip: could not invoke rng-diagnostics ({exc})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=Path, default=Path("seed.yaml"))
    args = p.parse_args()
    seed = load_seed(args.seed)
    reproducibility_check(seed)
    try_rng_diagnostics(seed)


if __name__ == "__main__":
    main()
