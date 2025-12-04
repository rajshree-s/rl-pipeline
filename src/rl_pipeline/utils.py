import os
from pathlib import Path


def get_project_root() -> Path:
    current_file_path = os.path.abspath(__file__)
    return Path(current_file_path).parent.parent.parent


def dataset_cache_dir() -> Path:
    p = os.getenv("DATASET_CACHE_DIR")
    if (
        not p
        or (not (path := Path(p)).exists())
        or (path.is_dir() and os.access(path, os.W_OK))
    ):
        path = get_project_root() / "datasets"
        print(f"Falling back to '{path.as_uri()}' for DATASET_CACHE_DIR")
    return path
