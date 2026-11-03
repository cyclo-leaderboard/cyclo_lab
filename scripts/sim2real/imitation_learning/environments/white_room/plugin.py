"""Local WhiteRoom scene plugin.  Safe to import before AppLauncher starts."""

from __future__ import annotations

import importlib
from pathlib import Path


ENVIRONMENT = "white_room"
_ROOT = Path(__file__).resolve().parent
_RUNTIME = _ROOT / "runtime"


def configure_launcher_args(args) -> None:
    args.enable_cameras = True


def _append_package_path(module_name: str, relative: str) -> None:
    module = importlib.import_module(module_name)
    candidate = str(_RUNTIME / relative)
    if candidate not in module.__path__:
        module.__path__.append(candidate)


def register_environments() -> None:
    _append_package_path("cyclo_lab.assets.environments", "cyclo_lab/assets/environments")
    _append_package_path(
        "cyclo_lab.manager_based.manipulation",
        "cyclo_lab/manager_based/manipulation",
    )
    importlib.import_module(
        "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    )


def provenance() -> dict[str, str]:
    return {"name": ENVIRONMENT, "layout_contract": "local_white_room_v1"}
