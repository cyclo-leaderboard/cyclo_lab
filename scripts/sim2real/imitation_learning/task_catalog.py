"""Task-number lookup for registered Cyclo Lab environments.

This module reads tracked task manifests without importing Isaac Sim. Local
plugins and assets can stay ignored while task-number mappings remain portable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_REPO = Path(__file__).resolve().parents[3]
_CATALOG_TASKS = Path(__file__).resolve().parent / "tasks/catalog"
_SOURCE_TASKS = (
    _REPO / "source/cyclo_lab/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks"
)
_TASK_ID = re.compile(r"(?:\d{6}|\d{3,6}_[a-z][a-z0-9_]*)\Z")
_SCENE_NAME = {"showroom": "Showroom", "white_room": "WhiteRoom", "wall_room": "WallRoom"}


def load_catalog() -> dict[str, dict]:
    """Load and validate every task-number manifest in this checkout."""
    paths = sorted(_CATALOG_TASKS.glob("task_*/task.json"))
    paths += sorted(_SOURCE_TASKS.rglob("task_*/task.json"))
    catalog: dict[str, dict] = {}
    env_owners: dict[str, str] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        task_id = data.get("task_id")
        if data.get("schema_version") != 1 or not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise ValueError(f"Invalid task manifest identity: {path}")
        if path.parent.name != f"task_{task_id}" or task_id in catalog:
            raise ValueError(f"Duplicate or misplaced task {task_id}: {path}")
        scene = data.get("environment")
        if scene not in _SCENE_NAME:
            raise ValueError(f"Unknown environment for task {task_id}: {scene!r}")
        if data.get("robot") not in {"FFW-SG2", "FFW-BG2"}:
            raise ValueError(f"Unknown robot for task {task_id}: {data.get('robot')!r}")
        if data.get("kind") not in {"task", "environment_preview"}:
            raise ValueError(f"Unknown task kind: {path}")
        if data.get("status") not in (None, "setup"):
            raise ValueError(f"Unknown task status: {path}")
        variants = data.get("variants")
        if not isinstance(variants, dict) or "default" in variants:
            raise ValueError(f"Invalid variants for task {task_id}: {path}")
        for name, env_id in {"default": data.get("default_env_id"), **variants}.items():
            if not isinstance(name, str) or not isinstance(env_id, str):
                raise ValueError(f"Invalid environment ID in {path}")
            if f"-Task{task_id}-" not in env_id:
                raise ValueError(f"Environment ID does not match task {task_id}: {env_id}")
            if not env_id.startswith(f"Cyclo-Real-{_SCENE_NAME[scene]}-"):
                raise ValueError(f"Environment family does not match task {task_id}: {env_id}")
            if env_id in env_owners:
                raise ValueError(f"Environment ID {env_id} is used by both {env_owners[env_id]} and {task_id}")
            env_owners[env_id] = task_id
        data["manifest_path"] = str(path)
        catalog[task_id] = data
    for task_id, data in catalog.items():
        base_task = data.get("base_task")
        if base_task is not None and base_task not in catalog:
            raise ValueError(f"Task {task_id} references missing base task {base_task}")
    return catalog


def validate_catalog() -> dict[str, int]:
    """Check tracked manifests and any available local plugins without Isaac Sim."""
    from local_plugins import load_for_task

    catalog = load_catalog()
    counts = {"task": 0, "environment_preview": 0, "setup": 0}
    for task_id, task in catalog.items():
        counts[task["kind"]] += 1
        counts["setup"] += task.get("status") == "setup"
        task_dir = Path(task["manifest_path"]).parent
        if _CATALOG_TASKS in task_dir.parents:
            plugin_path = task_dir / "plugin.py"
            if not plugin_path.is_file():
                raise ValueError(f"Task {task_id} has no tracked plugin.py")
            bundle = load_for_task(task["default_env_id"])
            if bundle.task_path != plugin_path or bundle.task_id != task_id or bundle.environment_key != task["environment"]:
                raise ValueError(f"Task {task_id} does not resolve to its tracked plugin")
        elif not (task_dir / "env_cfg.py").is_file() or not (task_dir / "spec.py").is_file():
            raise ValueError(f"Tracked task {task_id} lacks env_cfg.py or spec.py")
    return counts


def is_task_key(value: str) -> bool:
    """Recognize a catalog key, optionally prefixed with task_."""
    return bool(_TASK_ID.fullmatch(value.removeprefix("task_")))


def resolve_task(task_id: str, variant: str = "default") -> str:
    """Return the Gym ID for a task number and optional variant."""
    task_id = task_id.removeprefix("task_")
    catalog = load_catalog()
    if task_id not in catalog:
        raise KeyError(f"Unknown task number {task_id}")
    task = catalog[task_id]
    if variant == "default":
        return task["default_env_id"]
    try:
        return task["variants"][variant]
    except KeyError as exc:
        choices = ", ".join(("default", *sorted(task["variants"])))
        raise KeyError(f"Task {task_id} has no {variant!r} variant; choose {choices}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve task numbers to Gym environment IDs")
    parser.add_argument("task_id", nargs="?", help="Six-digit task number or named derivative such as 901_haeun")
    parser.add_argument("--variant", default="default", help="Environment variant")
    parser.add_argument("--list", action="store_true", help="List all task numbers")
    parser.add_argument("--validate", action="store_true", help="Check all task packages without Isaac Sim")
    args = parser.parse_args()
    if args.validate:
        counts = validate_catalog()
        print(f"Validated {counts['task']} tasks ({counts['setup']} in setup) and {counts['environment_preview']} previews")
        return
    if args.list:
        for task_id, task in sorted(load_catalog().items()):
            print(task_id, task["kind"], task.get("status", "registered"), task["environment"], task["robot"], task["default_env_id"])
        return
    if not args.task_id:
        parser.error("task_id is required unless --list is used")
    try:
        print(resolve_task(args.task_id, args.variant))
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
