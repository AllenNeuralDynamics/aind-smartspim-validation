"""
Utility functions
"""

from .types import PathLike


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.
    Parameters
    ------------------------
    txt: str
        String to be saved.
    filepath: PathLike
        Path where the file is located or will be saved.
    mode: str
        File open mode.
    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")
