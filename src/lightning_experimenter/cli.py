"""Command line entry point: `lexp materialize | run | train | status`."""

import argparse
import json
from pathlib import Path

from .dispatch import is_complete, pending, run_local, slurm_config, submit_slurm
from .materialize import materialize, read_manifest
from .train import train_from_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexp")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("materialize", help="expand a sweep into a run directory")
    p.add_argument("sweep", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, required=True)

    p = sub.add_parser("run", help="execute the configs in a run directory")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--backend", choices=["local", "slurm"], default="local")
    p.add_argument("--only-missing", action="store_true", help="skip completed runs")
    p.add_argument("--throttle", type=int, default=None, help="max concurrent array tasks")
    p.add_argument("--dry-run", action="store_true", help="write submit.sh without sbatch")

    p = sub.add_parser("train", help="train a single config (the array task payload)")
    p.add_argument("config", type=Path)

    p = sub.add_parser("status", help="report completion across a run directory")
    p.add_argument("out_dir", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "materialize":
        manifest = materialize(args.sweep, args.out_dir)
        print(f"{len(read_manifest(args.out_dir))} configs -> {manifest}")

    elif args.command == "train":
        print(json.dumps(train_from_file(args.config), indent=2))

    elif args.command == "run":
        entries = read_manifest(args.out_dir)
        todo = pending(entries) if args.only_missing else entries
        if not todo:
            print("nothing to do — all runs complete")
            return 0
        if args.backend == "local":
            run_local(todo)
        else:
            submit_slurm(
                args.out_dir,
                todo,
                slurm_config(args.out_dir),
                throttle=args.throttle,
                dry_run=args.dry_run,
            )

    elif args.command == "status":
        entries = read_manifest(args.out_dir)
        done = [e for e in entries if is_complete(e)]
        print(f"{len(done)}/{len(entries)} complete")
        for entry in entries:
            if not is_complete(entry):
                print(f"  missing {entry['index']:04d}  {entry['version_tag']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
