"""WallRoom scene selection for BG2 tasks."""

ENVIRONMENT = "wall_room"


def configure_launcher_args(args) -> None:
    args.enable_cameras = True


def register_environments() -> None:
    # Task plugins register the task-specific Gym variants.
    from cyclo_lab.assets.environments import wall_room  # noqa: F401


def provenance() -> dict[str, str]:
    return {"name": ENVIRONMENT, "layout_contract": "bg2_wall_room_v1"}
