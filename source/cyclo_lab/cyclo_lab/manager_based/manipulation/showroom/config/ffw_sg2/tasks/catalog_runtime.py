"""Connect tracked task packages to their existing cyclo_lab import names."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path


def catalog_root() -> Path:
    override = os.environ.get("CYCLO_LAB_TASK_CATALOG_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts/sim2real/imitation_learning/tasks/catalog"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Cyclo Lab task catalog is unavailable; set CYCLO_LAB_TASK_CATALOG_ROOT")


def mount_task_package(
    task_id: str, package_name: str, relative: str, *, required: bool = True
) -> None:
    """Expose a catalog runtime directory through an existing Python package."""
    package = importlib.import_module(package_name)
    path = catalog_root() / f"task_{task_id}" / "runtime" / relative
    if not path.is_dir():
        if required:
            raise RuntimeError(f"Task {task_id} runtime package is missing: {path}")
        return
    candidate = str(path)
    if candidate not in package.__path__:
        package.__path__.append(candidate)


def register_catalog_task(task_id: str) -> None:
    """Load the task-owned Gym registration during legacy package import."""
    path = catalog_root() / f"task_{task_id}" / "plugin.py"
    if not path.is_file():
        raise RuntimeError(f"Task {task_id} plugin is missing: {path}")
    spec = importlib.util.spec_from_file_location(f"_cyclo_catalog_task_{task_id}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load task {task_id} plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_environments()


def register_scene_dependency(scene_name: str) -> None:
    """Load a shared scene used by a task inherited from another scene family."""
    path = catalog_root().parent.parent / "environments" / scene_name / "plugin.py"
    if not path.is_file():
        raise RuntimeError(f"Shared scene plugin is missing: {path}")
    spec = importlib.util.spec_from_file_location(f"_cyclo_scene_{scene_name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared scene plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_environments()
