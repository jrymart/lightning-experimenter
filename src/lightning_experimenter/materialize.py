"""Turn a sweep file into a run directory: N configs plus a manifest.

Anything shared across the whole sweep — computing split statistics, for
instance — happens here, once, on the submit side. If it ran inside each task
the array would race on writing the same file.
"""

import json
import shutil
from pathlib import Path

import yaml

from .configs import build_configs, run_dir_for, set_path
from .resolve import load_callable

MANIFEST_FILE = "manifest.jsonl"


def materialize(sweep_path: Path, out_dir: Path) -> Path:
    sweep_path, out_dir = Path(sweep_path), Path(out_dir)
    sweep = yaml.safe_load(sweep_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_path, out_dir / "sweep.yaml")  # provenance

    # Absolute log_dir so array tasks don't depend on their working directory.
    sweep["fixed"].setdefault("train", {})
    sweep["fixed"]["train"].setdefault("log_dir", str((out_dir / "tb_logs").resolve()))

    if spec := sweep.get("prepare"):
        patches = load_callable(spec)(sweep, out_dir) or {}
        for dotted, value in patches.items():
            set_path(sweep["fixed"], dotted, value)

    configs = build_configs(sweep)
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    manifest = out_dir / MANIFEST_FILE
    with manifest.open("w") as handle:
        for index, run in enumerate(configs):
            config_path = config_dir / f"{index:04d}.yaml"
            yaml.safe_dump(run, config_path.open("w"), sort_keys=False)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "config": str(config_path.resolve()),
                        "run_dir": str(run_dir_for(run).resolve()),
                        "name_tag": run["name_tag"],
                        "version_tag": run["version_tag"],
                        "axes": run["axes"],
                    }
                )
                + "\n"
            )
    return manifest


def read_manifest(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path} — run `lexp materialize` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
