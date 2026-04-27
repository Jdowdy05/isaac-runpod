from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, resources

import gymnasium as gym


@dataclass(frozen=True)
class TeleopTaskSpec:
    env_cfg_entry_point: str
    rsl_rl_cfg_entry_point: str | None = None
    rl_games_cfg_entry_point: str | None = None
    add_cfg_entry_point: str | None = None
    task_slug: str | None = None


def _split_entry_point(entry_point: str) -> tuple[str, str]:
    if ":" in entry_point:
        module_name, attr_name = entry_point.split(":", 1)
        if not module_name or not attr_name:
            raise ValueError(f"Invalid entry point: {entry_point!r}")
        return module_name, attr_name
    module_name, _, attr_name = entry_point.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid entry point: {entry_point!r}")
    return module_name, attr_name


def load_object_from_entry_point(entry_point: str):
    module_name, attr_name = _split_entry_point(entry_point)
    module = import_module(module_name)
    return getattr(module, attr_name)


def resolve_resource_entry_point(entry_point: str) -> str:
    package_name, resource_name = _split_entry_point(entry_point)
    return str(resources.files(package_name).joinpath(resource_name))


def get_task_spec(task_name: str) -> TeleopTaskSpec:
    spec = gym.spec(task_name)
    kwargs = spec.kwargs
    return TeleopTaskSpec(
        env_cfg_entry_point=kwargs["env_cfg_entry_point"],
        rsl_rl_cfg_entry_point=kwargs.get("rsl_rl_cfg_entry_point"),
        rl_games_cfg_entry_point=kwargs.get("rl_games_cfg_entry_point"),
        add_cfg_entry_point=kwargs.get("add_cfg_entry_point"),
        task_slug=kwargs.get("task_slug"),
    )


def make_env_cfg_for_task(task_name: str):
    cfg_type = load_object_from_entry_point(get_task_spec(task_name).env_cfg_entry_point)
    return cfg_type()


def make_rsl_runner_cfg_for_task(task_name: str):
    entry_point = get_task_spec(task_name).rsl_rl_cfg_entry_point
    if entry_point is None:
        raise ValueError(f"Task {task_name!r} does not define an RSL-RL runner config entry point.")
    cfg_type = load_object_from_entry_point(entry_point)
    return cfg_type()


def resolve_add_config_path_for_task(task_name: str) -> str:
    entry_point = get_task_spec(task_name).add_cfg_entry_point
    if entry_point is None:
        raise ValueError(f"Task {task_name!r} does not define an ADD config entry point.")
    return resolve_resource_entry_point(entry_point)


def task_slug_for_task(task_name: str) -> str:
    task_slug = get_task_spec(task_name).task_slug
    if task_slug:
        return task_slug
    slug = task_name
    if slug.startswith("Isaac-"):
        slug = slug[len("Isaac-") :]
    for suffix in ("-Direct-v0", "-v0"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug.replace("-", "_").lower()
