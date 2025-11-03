from __future__ import annotations
import yaml, pathlib, random, numpy as np

def load_config(path="config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42):
    import os, torch  # torch optional; handle gracefully
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(p: str | pathlib.Path):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
