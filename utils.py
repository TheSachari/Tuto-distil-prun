import json, gzip, shutil
from pathlib import Path
import random
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_state_dict_size_bytes(model):
    return sum(p.numel()*p.element_size() for p in model.state_dict().values())


def save_state_dict(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def save_and_gzip(model, path: Path):
    raw = save_state_dict(model, path)
    gz_path = raw.with_suffix(raw.suffix + ".gz")
    with open(raw, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return raw, gz_path


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
