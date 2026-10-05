#!/usr/bin/env python3
"""Inventory only: no model download, inference, package installation, or secrets."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes",
            "numpy", "scipy", "Pillow", "datasets", "pytest")

def inventory():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = {"available": False, "details": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=False)
            gpu = {"available": result.returncode == 0,
                   "details": result.stdout.strip() if result.returncode == 0 else None,
                   "error": result.stderr.strip() if result.returncode else None}
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu["error"] = type(exc).__name__
    return {
        "kind": "environment_inventory_not_compatibility_test",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_at_least_3_10": sys.version_info >= (3, 10),
        "packages": versions,
        "nvidia_smi": gpu,
        "model_loaded": False,
        "dataset_checked": False,
        "quantization_audited": False,
        "ready_for_main_run": False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = json.dumps(inventory(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")

if __name__ == "__main__":
    main()
