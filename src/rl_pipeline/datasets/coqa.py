from pathlib import Path
from rl_pipeline.utils import get_project_root

from datasets import load_dataset


def _get_download_path() -> Path:
    return get_project_root() / "datasets"


def download():
    load_dataset("stanfordnlp/coqa", cache_dir=_get_download_path().as_uri())
