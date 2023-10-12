"""
Dataset reader
"""

import os
import re
from enum import Enum
from pathlib import Path
from typing import List

from .types import PathLike


class SmartSPIMReader:
    """Reader for smartspim datasets"""

    class RegexPatterns(Enum):
        """Enum for regex patterns for the smartSPIM data"""

        # regex expressions for not structured smartspim datasets
        capture_date_regex = r"(20[0-9]{2}([0-9][0-9]{1})([0-9][0-9]{1}))"
        capture_time_regex = r"(_(\d{2})_(\d{2})_(\d{2})_)"
        capture_mouse_id = r"(_(\d+|[a-zA-Z]*\d+)$)"

        # Regular expression for smartspim datasets
        smartspim_regex = r"SmartSPIM_(\d+|[a-zA-Z]*\d+)_(20\d{2}-(\d\d{1})-(\d\d{1}))_((\d{2})-(\d{2})-(\d{2}))"
        smartspim_old_regex = r"(20\d{2}(\d\d{1})(\d\d{1}))_((\d{2}))_((\d{2}))_((\d{2}))_(\d+|[a-zA-Z]*\d+)"

        # Regex expressions for inner folders inside root
        regex_channels = r"Ex_(\d{3})_Em_(\d{3})$"
        regex_channels_MIP = r"Ex_(\d{3})_Em_(\d{3}_MIP)$"
        regex_files = r'[^"]*.(txt|ini)$'

    @staticmethod
    def read_smartspim_folders(path: PathLike) -> List[str]:
        """
        Reads smartspim datasets in a folder
        based on data conventions

        Parameters
        -----------------
        path: PathLike
            Path where the datasets are located

        Returns
        -----------------
        List[str]
            List with the found smartspim datasets
        """
        smartspim_datasets = []

        if os.path.isdir(path):
            datasets = os.listdir(path)

            for dataset in datasets:
                dataconvention_match = re.match(
                    SmartSPIMReader.RegexPatterns.smartspim_regex.value,
                    dataset,
                )

                oldconvention_match = re.match(
                    SmartSPIMReader.RegexPatterns.smartspim_old_regex.value,
                    dataset,
                )

                if dataconvention_match:
                    str_path = str(Path(dataset).joinpath(POST_PATH_NEW_CONV))
                    smartspim_datasets.append(str_path)

                if oldconvention_match:
                    str_path = str(Path(dataset).joinpath(POST_PATH_OLD_CONV))
                    smartspim_datasets.append(str_path)

        else:
            raise ValueError(f"Path {path} is not a folder.")

        return smartspim_datasets
