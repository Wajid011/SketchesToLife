"""Unit tests for sketch2image.utils — config loading, path resolution, seed."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from sketch2image.utils import _deep_merge, load_config, resolve_output_dir, set_seed

# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_basic():
    base = {"a": 1, "b": 2}
    overlay = {"b": 99, "c": 3}
    result = _deep_merge(base, overlay)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    base = {"models": {"sd15": "old-id", "cn": "cn-id"}, "seed": 42}
    overlay = {"models": {"sd15": "new-id"}, "seed": 7}
    result = _deep_merge(base, overlay)
    assert result["models"]["sd15"] == "new-id"
    assert result["models"]["cn"] == "cn-id"  # unchanged
    assert result["seed"] == 7


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    overlay = {"a": {"x": 99}}
    _deep_merge(base, overlay)
    assert base["a"]["x"] == 1  # original unchanged


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_with_tmp_yaml(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(yaml.dump({"seed": 999, "guidance_scale": 8.5}))

    # Patch the default config path to avoid depending on disk layout
    with patch(
        "sketch2image.utils._DEFAULT_CONFIG_PATH", tmp_path / "nonexistent.yaml"
    ):
        config = load_config(cfg_file)

    assert config["seed"] == 999
    assert config["guidance_scale"] == 8.5


def test_load_config_overrides(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(yaml.dump({"seed": 42}))

    with patch(
        "sketch2image.utils._DEFAULT_CONFIG_PATH", tmp_path / "nonexistent.yaml"
    ):
        config = load_config(cfg_file, overrides={"seed": 123})

    assert config["seed"] == 123


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/absolutely/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# resolve_output_dir
# ---------------------------------------------------------------------------


def test_resolve_output_dir_from_config(tmp_path):
    config = {"output_dir": str(tmp_path / "my_outputs")}
    out = resolve_output_dir(config)
    assert out.exists()


def test_resolve_output_dir_env_var_wins(tmp_path):
    env_path = tmp_path / "env_outputs"
    with patch.dict(os.environ, {"OUTPUT_DIR": str(env_path)}):
        config = {"output_dir": str(tmp_path / "config_outputs")}
        out = resolve_output_dir(config)
    assert str(out) == str(env_path.resolve())
    assert out.exists()


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


def test_set_seed_runs_without_torch():
    """set_seed must not crash even when torch is unavailable."""
    import sys

    with patch.dict(sys.modules, {"torch": None}):
        set_seed(42)  # should not raise


def test_set_seed_deterministic():
    import random

    set_seed(0)
    a = random.random()
    set_seed(0)
    b = random.random()
    assert a == b
