import pytest

from local_code_rag.cli import main
from local_code_rag.config import load_config


def test_load_config_defaults_and_relative_paths(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    (tmp_path / "config.toml").write_text('[repo]\npath = "proj"\n\n[index]\ndir = "idx"\n')
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.repo_path == repo.resolve()
    assert cfg.repo_name == "proj"
    assert cfg.index_dir == tmp_path.resolve() / "idx"
    assert cfg.embed_model == "nomic-embed-text"
    assert ".py" in cfg.extensions


def test_load_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
    (tmp_path / "c.toml").write_text("[repo]\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "c.toml")


def test_cli_index_and_search(tmp_path, sample_repo, capsys):
    conf = tmp_path / "c.toml"
    conf.write_text(f'[repo]\npath = "{sample_repo}"\nname = "s"\n[ollama]\nembed_model = "hashing"\n'
                    f'[index]\ndir = "{tmp_path / "idx"}"\n')
    main(["--config", str(conf), "index"])
    main(["--config", str(conf), "search", "who calls write_table"])
    out = capsys.readouterr().out
    assert "load_orders_silver" in out and "pipelines/orders.py" in out
    main(["--config", str(conf), "stats"])
    assert "symbols" in capsys.readouterr().out
