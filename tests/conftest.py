from pathlib import Path

import pytest

from local_code_rag.config import Config

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_repo"


@pytest.fixture
def sample_repo() -> Path:
    return SAMPLE


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(repo_path=SAMPLE, repo_name="sample-shop", embed_model="hashing", index_dir=tmp_path / "idx")
