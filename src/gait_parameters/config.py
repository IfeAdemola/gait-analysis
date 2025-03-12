from pathlib import Path
from my_utils.helpers import get_file_root, get_repo_root, get_project_root

MAIN_ROOT = get_file_root(__file__)
REPO_ROOT = get_repo_root(MAIN_ROOT)
PROJECT_ROOT = get_project_root(MAIN_ROOT)