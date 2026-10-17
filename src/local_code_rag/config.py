"""Configuration: one TOML file, sensible defaults for everything but the repo path."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXTENSIONS = [".py", ".sql", ".yaml", ".yml", ".md"]
DEFAULT_EXCLUDES = [".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build", ".mypy_cache"]


@dataclass
class Config:
    repo_path: Path
    repo_name: str
    extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"   # or "hashing" to run fully offline without Ollama
    index_dir: Path | None = None
    max_results: int = 10
    project_description: str = ""
    extra_read_patterns: list[str] = field(default_factory=list)
    extra_write_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path).expanduser().resolve()
        if self.index_dir is None:
            self.index_dir = Path("~/.local-code-rag").expanduser() / self.repo_name
        self.index_dir = Path(self.index_dir).expanduser()

    @property
    def db_path(self) -> Path:
        return self.index_dir / "symbols.db"

    @property
    def vector_path(self) -> Path:
        return self.index_dir / "vectors"


def load_config(path: str | Path = "config.toml") -> Config:
    """Load a config file. Only `[repo] path` is required."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Copy config.example.toml to config.toml and set [repo] path."
        )
    with p.open("rb") as f:
        raw = tomllib.load(f)

    repo = raw.get("repo", {})
    if "path" not in repo:
        raise ValueError("config: [repo] path is required")
    ollama = raw.get("ollama", {})
    index = raw.get("index", {})
    tables = raw.get("tables", {})
    # Relative paths are resolved against the config file's folder, not the current directory.
    base = p.resolve().parent
    repo_path = base / Path(repo["path"]).expanduser()
    index_dir = base / Path(index["dir"]).expanduser() if index.get("dir") else None

    return Config(
        repo_path=repo_path,
        repo_name=repo.get("name") or repo_path.resolve().name,
        extensions=repo.get("extensions", list(DEFAULT_EXTENSIONS)),
        exclude_dirs=repo.get("exclude_dirs", list(DEFAULT_EXCLUDES)),
        ollama_url=ollama.get("base_url", "http://localhost:11434"),
        chat_model=ollama.get("chat_model", "qwen2.5-coder:7b"),
        embed_model=ollama.get("embed_model", "nomic-embed-text"),
        index_dir=index_dir,
        max_results=raw.get("ui", {}).get("max_results", 10),
        project_description=repo.get("description", ""),
        extra_read_patterns=tables.get("read_patterns", []),
        extra_write_patterns=tables.get("write_patterns", []),
    )
