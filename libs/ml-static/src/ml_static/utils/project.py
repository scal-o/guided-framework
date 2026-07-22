from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root by searching upwards for a marker file.

    The project root is identified by the presence of a 'pyproject.toml' file,
    which is standard for modern Python projects.

    Returns:
        The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the project root marker is not found.
    """
    # Start from the directory of the current file
    current_path = Path(__file__).resolve().parent
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent

    # As a fallback for the case where the loop terminates at the root
    # without finding the file (e.g., '/'), check the final path as well.
    if (current_path / "pyproject.toml").exists():
        return current_path

    raise FileNotFoundError(
        "Could not find project root. The 'pyproject.toml' file was not found in any parent directory."
    )
