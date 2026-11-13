import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def seed_all(seed=42, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def rng_state(device):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device == "mps":
        torch.mps.set_rng_state(state["mps"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def select_device(requested="auto"):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Use an Apple Silicon Mac or --device cpu."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    return requested
