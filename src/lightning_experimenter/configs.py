"""Expand a sweep config (`experiment` axes over a `fixed` tree) into run configs."""

from copy import deepcopy
from itertools import product
from pathlib import Path


def set_path(cfg: dict, dotted: str, value) -> None:
    """Assign `value` at a dotted path, erroring if a parent section is missing."""
    *parents, leaf = dotted.split(".")
    node = cfg
    for depth, key in enumerate(parents):
        if not isinstance(node, dict) or key not in node:
            prefix = ".".join(parents[: depth + 1])
            raise KeyError(
                f"sweep axis {dotted!r} targets section {prefix!r}, "
                "which does not exist in the fixed config"
            )
        node = node[key]
    node[leaf] = value


def name_fragment(axis_name: str, tag: str) -> str:
    leaf = axis_name.split(".")[-1]
    return tag if leaf == axis_name else f"{leaf}{tag}"


def axis_items(name: str, values) -> list[tuple[str, dict]]:
    """Normalise an axis into (tag, assignments) pairs.

    A dict axis maps its own tags to explicit dotted assignments; a list axis
    assigns each value to the axis name itself.
    """
    if isinstance(values, dict):
        return list(values.items())
    return [(str(v), {name: v}) for v in values]


def run_dir_for(cfg: dict) -> Path:
    """Where a run writes its logs, checkpoint and metrics.

    Must match TensorBoardLogger.log_dir exactly (save_dir/name/version) so the
    manifest can record it without instantiating a logger.
    """
    return Path(cfg["train"].get("log_dir", "tb_logs")) / cfg["name_tag"] / cfg["version_tag"]


def build_configs(cfg: dict) -> list[dict]:
    """Cartesian product over `cfg['experiment']`, applied to `cfg['fixed']`."""
    axes = {name: axis_items(name, values) for name, values in cfg["experiment"].items()}

    name_axis = cfg.get("name_tag")
    if name_axis is not None and name_axis not in axes:
        raise KeyError(
            f"name_tag {name_axis!r} is not one of the experiment axes: {sorted(axes)}"
        )

    configs: list[dict] = []
    seen: set[str] = set()
    for combo in product(*axes.values()):
        run = deepcopy(cfg["fixed"])
        for _, assignments in combo:
            for path, value in assignments.items():
                set_path(run, path, value)

        tags = {name: tag for name, (tag, _) in zip(axes, combo)}
        # For a list axis keep the real value so hparams stay numeric and
        # sortable; a dict axis has no single value, so its tag stands in.
        coords = {
            name: assignments.get(name, tag)
            for name, (tag, assignments) in zip(axes, combo)
        }
        version = "__".join(name_fragment(name, tag) for name, tag in tags.items())
        if version in seen:
            raise ValueError(
                f"duplicate version_tag {version!r} — two axis values produce the "
                "same name fragment, so runs would overwrite each other"
            )
        seen.add(version)

        run["version_tag"] = version
        run["name_tag"] = tags.get(name_axis, "unnamed_experiment")
        run["axes"] = coords  # flat coordinates, logged as TensorBoard hparams
        configs.append(run)
    return configs
