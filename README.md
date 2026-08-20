# lightning-experimenter

Config-driven PyTorch Lightning harness. A sweep file expands into N run configs;
each run is trained by a single command, so the same configs execute in a local
loop or as a SLURM array.

## Usage

```bash
lexp materialize sweep.yaml -o runs/2026-08-19/   # expand + one-time prep
lexp run runs/2026-08-19/                         # local loop
lexp run runs/2026-08-19/ --backend slurm --throttle 8
lexp status runs/2026-08-19/
lexp run runs/2026-08-19/ --backend slurm --only-missing   # resubmit failures
lexp train runs/2026-08-19/configs/0042.yaml      # one run; the array payload
```

## Run directory

```
runs/2026-08-19/
  sweep.yaml            verbatim copy of the input
  manifest.jsonl        index -> config path, run dir, axis coordinates
  configs/0000.yaml     one per grid point, zero-padded to match array index
  submit.sh             generated sbatch script
  logs/%A_%a.{out,err}
  tb_logs/<name_tag>/<version_tag>/
      config.yaml  best.ckpt  metrics.json  events.out.tfevents.*
```

`metrics.json` is written last, so its presence is what `--only-missing` treats
as "complete". `best.ckpt` would be wrong: it appears at the first validation, so
a preempted job would look finished.

## Sweep file

```yaml
name_tag: inputs          # which axis becomes the TensorBoard `name` folder
prepare: gw_signatures.stats.compute_split_stats   # optional, runs once
slurm:                    # any key here becomes an #SBATCH directive
  partition: blanca-csdms
  time: "04:00:00"
  gres: gpu:1

experiment:               # cartesian product of these axes
  inputs:                 # dict axis: named variants, explicit assignments
    dem_only:
      data.data_types: [dem]
      data.scalars: {}
  data.patch_size: [64, 128]     # list axis: assigns to its own dotted path
  train.seed: [10, 20, 30]

fixed:                    # base tree the axes are applied to
  data:
    class: gw_signatures.data.PatchBagDataModule
    batch_size: 16
  model:
    class: gw_signatures.models.TerrainCNN
    kernel_size: 3
  train:
    monitor: val/basin_mse
    max_steps: 20000
    val_every_n_steps: 500
```

Exactly one of `train.max_steps` / `train.max_epochs` must be set.

## The prepare hook

Runs once, before expansion, for work shared by the whole sweep. Returns dotted
patches merged into `fixed`:

```python
def compute_split_stats(sweep: dict, out_dir: Path) -> dict:
    stats_path = out_dir / "stats.yaml"
    ...
    return {"data.stats_path": str(stats_path.resolve())}
```

Returning patches rather than mutating means paths land in every config as
absolute, so array tasks don't depend on their working directory.

## Module contract

`data.class` and `model.class` are dotted paths to a `LightningDataModule` and a
`LightningModule`; both are constructed with the full run config, `Cls(cfg)`.
Nothing else about your project is known to this package.
