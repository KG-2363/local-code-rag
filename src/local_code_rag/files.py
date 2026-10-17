"""Walk a repository, respecting excluded directories."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def iter_files(repo_root: Path, extensions: list[str], exclude_dirs: list[str]) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, POSIX path relative to the repo) for matching files, sorted."""
    exts = {e.lower() for e in extensions}
    excluded = set(exclude_dirs)
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel_parts = path.relative_to(repo_root).parts
        if any(part in excluded for part in rel_parts[:-1]):
            continue
        yield path, "/".join(rel_parts)
