from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from graph_pes.config.training import SWAConfig, TrainingConfig
from graph_pes.models import load_model
from graph_pes.scripts.train import train_from_config
from graph_pes.scripts.utils import extract_config_dict_from_command_line
from graph_pes.training.callbacks import WandbLogger
from graph_pes.utils.misc import nested_merge

from .. import helpers


def test_arg_parse():
    config_path = helpers.CONFIGS_DIR / "minimal.yaml"
    command = f"""\
graph-pes-train {config_path} \
    fitting/loader_kwargs/batch_size=32 \
    data/+load_atoms_dataset/n_train=10
"""
    sys.argv = command.split()

    config_data = nested_merge(
        TrainingConfig.defaults(),
        extract_config_dict_from_command_line(""),
    )
    assert config_data["fitting"]["loader_kwargs"]["batch_size"] == 32
    assert config_data["data"]["+load_atoms_dataset"]["n_train"] == 10


def test_train_script(tmp_path: Path):
    root = tmp_path / "root"
    config = _get_quick_train_config(root)

    train_from_config(config)

    assert root.exists()
    sub_dir = next(root.iterdir())
    assert (sub_dir / "model.pt").exists()


def test_training_reproducibility(tmp_path: Path):
    states = []
    metrics = []
    for run_id, seed in [("default", None), ("same", 42), ("different", 43)]:
        config = _get_quick_train_config(tmp_path)
        config["model"] = {
            "+SchNet": {"cutoff": 3.0, "channels": 8, "layers": 1}
        }
        config["general"]["run_id"] = run_id
        config["general"].pop("seed")
        if seed is not None:
            config["SEED"] = seed
            config["general"]["seed"] = "=/SEED"
        config["fitting"]["trainer_kwargs"]["max_epochs"] = 2
        config["fitting"]["trainer_kwargs"]["deterministic"] = True
        config_path = tmp_path / f"{run_id}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = subprocess.run(
            [sys.executable, "-m", "graph_pes.scripts.train", str(config_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        model = load_model(tmp_path / run_id / "model.pt")
        states.append(
            {
                k: v
                for k, v in model.state_dict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        summary = yaml.safe_load(
            (tmp_path / run_id / "summary.yaml").read_text()
        )
        metrics.append(
            {k: v for k, v in summary.items() if k.startswith("test/")}
        )

    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(
            states[0][key], states[1][key], rtol=0, atol=0
        )
    assert metrics[0]
    assert metrics[0] == metrics[1]
    assert any(
        not torch.equal(states[0][key], states[2][key]) for key in states[0]
    )
    assert metrics[0] != metrics[2]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_before_model_initialization(tmp_path: Path, monkeypatch, dtype):
    class InitializationComplete(Exception):
        pass

    def check_constructor(**kwargs):
        assert torch.get_default_dtype() == getattr(torch, dtype)
        raise InitializationComplete

    monkeypatch.setattr("graph_pes.models.SchNet", check_constructor)
    config = _get_quick_train_config(tmp_path)
    config["model"] = {"+SchNet": {}}
    config["DTYPE"] = dtype
    config["general"]["torch"]["dtype"] = "=/DTYPE"
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(
            torch.float64 if dtype == "float32" else torch.float32
        )
        with pytest.raises(InitializationComplete):
            train_from_config(config)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_run_id(tmp_path: Path):
    root = tmp_path / "root"

    # first round: train with no explicit run_id ...
    config = _get_quick_train_config(root)
    assert config["general"]["run_id"] is None

    train_from_config(config)

    # second round: train with an explicit run_id
    config = _get_quick_train_config(root)
    config["general"]["run_id"] = "explicit-id"
    train_from_config(config)
    assert (root / "explicit-id").exists()

    # third round: train with the same explicit run_id
    # and check that the collision is avoided
    config = _get_quick_train_config(root)
    config["general"]["run_id"] = "explicit-id"
    train_from_config(config)
    assert (root / "explicit-id-1").exists()


def test_swa(tmp_path: Path, caplog):
    root = tmp_path / "root"
    config = _get_quick_train_config(root)
    config["fitting"]["swa"] = SWAConfig(lr=0.1, start=1)
    config["fitting"]["trainer_kwargs"]["max_epochs"] = 3

    train_from_config(config)

    assert "SWA: starting SWA" in caplog.text


def _get_quick_train_config(root) -> dict:
    config_str = f"""\
general:
    root_dir: {root}
    run_id: null
wandb: null
loss: +PerAtomEnergyLoss()
model: 
    +LennardJones: {{cutoff: 3.0}}
data:
    +load_atoms_dataset:
        id: {helpers.CU_STRUCTURES_FILE}
        cutoff: 3.0
        n_train: 6
        n_valid: 2
        n_test: 2
fitting:
    trainer_kwargs:
        max_epochs: 1
        accelerator: cpu
        callbacks: []
    loader_kwargs:
        batch_size: 2
        num_workers: 0
        persistent_workers: false
"""
    return nested_merge(
        TrainingConfig.defaults(),
        yaml.safe_load(config_str),
    )


def test_wandb_logging(tmp_path: Path, caplog):
    import wandb

    # configure wandb to not actually log anything
    wandb.init = lambda *args, **kwargs: None

    logger = WandbLogger(
        tmp_path / "logging-name",
        project="test-project",
        log_epoch=False,
    )
    assert logger._id == "logging-name"
    assert logger.save_dir == str(tmp_path)
    assert logger._project == "test-project"
