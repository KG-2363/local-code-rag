"""Index YAML/TOML-style configuration as searchable dotted keys.

Pipelines often keep table names, schedules and feature flags in YAML. Flattening
every file to `dotted.key = value` rows (with line numbers) makes questions like
"where is the retention period set?" answerable with an exact lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..files import iter_files


@dataclass
class ConfigKey:
    key: str             # e.g. "jobs[0].schedule" or "tables.orders.retention_days"
    value: str
    file_path: str
    line_number: int


class _LineLoader(yaml.SafeLoader):
    """SafeLoader that records the line of every mapping key."""


def _construct_mapping(loader: _LineLoader, node: yaml.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        mapping[(key, key_node.start_mark.line + 1)] = loader.construct_object(value_node, deep=deep)
    return mapping


_LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def index_configs(repo_root: Path, exclude_dirs: list[str]) -> list[ConfigKey]:
    keys: list[ConfigKey] = []
    for path, rel in iter_files(repo_root, [".yaml", ".yml"], exclude_dirs):
        try:
            for doc in yaml.load_all(path.read_text(encoding="utf-8", errors="replace"), Loader=_LineLoader):
                keys.extend(_flatten(doc, rel, "", 1))
        except yaml.YAMLError:
            continue
    return keys


def _flatten(node, rel: str, prefix: str, line: int) -> list[ConfigKey]:
    out: list[ConfigKey] = []
    if isinstance(node, dict):
        for (k, k_line), v in node.items():
            out.extend(_flatten(v, rel, f"{prefix}.{k}" if prefix else str(k), k_line))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            out.extend(_flatten(item, rel, f"{prefix}[{i}]", line))
    elif prefix:
        out.append(ConfigKey(prefix, "" if node is None else str(node), rel, line))
    return out
