"""Load settings: configs/default.toml, overlaid with an experiment config, plus .env.

    config = load_config("configs/baseline.toml")

The experiment file only needs the keys it changes; everything else comes from
configs/default.toml, or from the file named by its `extends` key. Tables are
merged key by key (deep merge).

.env is read into os.environ (without overriding variables already set), so the
Anthropic SDK finds ANTHROPIC_API_KEY. This is ~10 lines instead of a dependency.
"""

import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path("configs/default.toml")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into os.environ. Skips blanks and # comments; never overrides."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def deep_merge(base: dict, override: dict) -> dict:
    """Return base with override's keys applied; nested tables are merged, not replaced."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_chain(path: Path) -> dict:
    """A config file, merged over its `extends = "..."` parent (or default.toml if it has none)."""
    data = tomllib.loads(path.read_text())
    parent = data.pop("extends", None)
    base = _load_chain(Path(parent)) if parent else tomllib.loads(DEFAULT_CONFIG.read_text())
    return deep_merge(base, data)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """default.toml, overlaid by `path`. A file may say `extends = "configs/baseline.toml"`
    to build on another config, so an experiment file holds only the one thing it changes."""
    load_dotenv()
    path = Path(path)
    if path == DEFAULT_CONFIG:
        return tomllib.loads(DEFAULT_CONFIG.read_text())
    return _load_chain(path)
