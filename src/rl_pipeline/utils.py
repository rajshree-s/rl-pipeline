import os
from pathlib import Path


def get_project_root() -> Path:
    current_file_path = os.path.abspath(__file__)
    return Path(current_file_path).parent.parent.parent
