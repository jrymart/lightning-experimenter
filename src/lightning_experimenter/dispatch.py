"""Two ways to execute the entries in a manifest: locally, or as a SLURM array."""

import subprocess
import sys
from pathlib import Path

import yaml

from .train import METRICS_FILE, train_from_file


def is_complete(entry: dict) -> bool:
    """A run counts as done once it has written metrics.json.

    Deliberately not `best.ckpt` — that appears at the first validation, so a
    job killed mid-run would look finished and never be resubmitted.
    """
    return (Path(entry["run_dir"]) / METRICS_FILE).exists()


def pending(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if not is_complete(entry)]


def run_local(entries: list[dict]) -> None:
    for entry in entries:
        print(f"[{entry['index']:04d}] {entry['name_tag']}/{entry['version_tag']}", flush=True)
        train_from_file(Path(entry["config"]))


def _array_spec(indices: list[int], throttle: int | None) -> str:
    """Compress indices into SLURM array syntax, collapsing runs into ranges."""
    if not indices:
        raise ValueError("nothing to submit")
    ordered = sorted(indices)
    parts, start, previous = [], ordered[0], ordered[0]
    for index in ordered[1:] + [None]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        if index is not None:
            start = previous = index
    spec = ",".join(parts)
    return f"{spec}%{throttle}" if throttle else spec


def write_sbatch(
    out_dir: Path,
    entries: list[dict],
    slurm_cfg: dict,
    throttle: int | None = None,
) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Every key in the sweep's `slurm:` block becomes an sbatch directive, so
    # switching clusters is a config change rather than a code change.
    directives = "\n".join(f"#SBATCH --{key}={value}" for key, value in slurm_cfg.items())

    script = f"""#!/bin/bash
#SBATCH --array={_array_spec([e["index"] for e in entries], throttle)}
#SBATCH --output={out_dir.resolve()}/logs/%A_%a.out
#SBATCH --error={out_dir.resolve()}/logs/%A_%a.err
{directives}
set -euo pipefail

CONFIG={out_dir.resolve()}/configs/$(printf "%04d" "$SLURM_ARRAY_TASK_ID").yaml
{sys.executable} -m lightning_experimenter.cli train "$CONFIG"
"""
    path = out_dir / "submit.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


def submit_slurm(
    out_dir: Path,
    entries: list[dict],
    slurm_cfg: dict,
    throttle: int | None = None,
    dry_run: bool = False,
) -> str | None:
    script = write_sbatch(out_dir, entries, slurm_cfg, throttle)
    print(f"wrote {script} ({len(entries)} tasks)")
    if dry_run:
        return None
    result = subprocess.run(
        ["sbatch", str(script)], capture_output=True, text=True, check=True
    )
    print(result.stdout.strip())
    return result.stdout.strip()


def slurm_config(out_dir: Path) -> dict:
    sweep = yaml.safe_load((Path(out_dir) / "sweep.yaml").read_text())
    return sweep.get("slurm", {})
