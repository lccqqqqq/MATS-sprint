# Implement parameter sweeps
import yaml
from pathlib import Path
from typing import Dict, Any
import itertools
import os
import wandb
from trainer import train


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


sweep_config_toy = {
    "standardize_method": ["per_token", "per_batch"],
    "lr": [1e-5, 2e-5],
    "sparsity_coeff": [0.03125, 0.0625],
}

default_config_toy = {
    "standardize_method": "per_token",
    "lr": 2e-5,
    "sparsity_coeff": 0.03125,
}

sweep_config_real = {
    "standardize_method": ["per_token", "per_batch", "plain"],
    "lr": [2e-5, 5e-5, 1e-4],
    "sparsity_coeff": [0.03125, 0.0175],
}

default_config_real = {
    "standardize_method": "per_token",
    "lr": 2e-5,
    "sparsity_coeff": 0.03125,
}

def create_sweep_config(
    sweep_config_dict: dict,
    default_config_file: str,
    sweep_name: str,
) -> dict:
    dir_path = Path(default_config_file).parent
    default_config = load_config(default_config_file)
    
    # sweeping standardize_method
    std_method = sweep_config_dict["standardize_method"]
    lr = sweep_config_dict["lr"]
    sparsity_coeff = sweep_config_dict["sparsity_coeff"]
    
    for i, (std_method, lr, sparsity_coeff) in enumerate(itertools.product(std_method, lr, sparsity_coeff)):
        sweep_config = default_config.copy()
        sweep_config["sae"]["standardize_method"] = std_method
        sweep_config["optimizer"]["lr"] = lr
        sweep_config["sae"]["sparsity_coeff"] = sparsity_coeff
        sweep_config["trainer"]["sparsity_loss_alpha"] = sparsity_coeff
        sweep_config["save_dir"] = f"output/{sweep_name}_{i}"
        
        # save the sweep config
        sweep_config_path = os.path.join(dir_path, f"{sweep_name}_{i}.yaml")
        with open(sweep_config_path, "w") as f:
            yaml.dump(sweep_config, f)
            


if __name__ == "__main__":
    create_sweep_config(sweep_config_toy, "/workspace/sprint/sae-diff/configs/toy/gated_sae_toy.yaml", "sweep_toy")
    create_sweep_config(sweep_config_real, "/workspace/sprint/sae-diff/configs/real/gated_sae_on_diff.yaml", "sweep_real")
    
    SWEEP_FOLDER = "/workspace/sprint/sae-diff/configs/toy"
    PROJECT_NAME = "SAE-diffing-toy"
    
    yaml_files = sorted(
        filename for filename in os.listdir(SWEEP_FOLDER)
        if filename.endswith(".yaml") and filename.startswith("sweep")
    )
    
    # import random
    # random.shuffle(yaml_files)
    # for filename in yaml_files:
    #     sweep_path = os.path.join(SWEEP_FOLDER, filename)
    #     print(f"Launching sweep from {filename}")
    #     cfg = load_config(sweep_path)
    #     monitor = train(cfg=cfg, use_monitor=True)
    #     monitor.plot()
    


