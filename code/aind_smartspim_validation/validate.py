"""
Script to validate smartspim datasets
"""

import logging
import multiprocessing
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import exiftool
from tqdm import tqdm

from .reader import SmartSPIMReader
from .types import PathLike

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_image_directory_structure(folder_dir: PathLike) -> dict:
    """
    Creates a dictionary representation of all the images
    saved by folder/col_N/row_N/images_N.[file_extention]

    Parameters
    ------------------------
    folder_dir:PathLike
        Path to the folder where the images are stored. This
        should point to the /SmartSPIM folder where the
        channels are stored.

    Returns
    ------------------------
    dict:
        Dictionary with the image representation where:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: integer with n images} } } }
    """

    directory_structure = {}
    folder_dir = Path(folder_dir)

    channel_paths = [
        folder_dir.joinpath(folder)
        for folder in os.listdir(folder_dir)
        if os.path.isdir(folder_dir.joinpath(folder))
        and re.match(SmartSPIMReader.RegexPatterns.regex_channels.value, folder)
    ]

    for channel_idx in range(len(channel_paths)):
        directory_structure[channel_paths[channel_idx]] = {}
        cols = os.listdir(channel_paths[channel_idx])
        # logger.info(f"Validating channel {channel_paths[channel_idx]}")

        for col in tqdm(cols):
            possible_col = channel_paths[channel_idx].joinpath(col)

            if os.path.isdir(possible_col):
                directory_structure[channel_paths[channel_idx]][col] = {}
                rows = os.listdir(possible_col)

                for row in rows:
                    possible_row = (
                        channel_paths[channel_idx].joinpath(col).joinpath(row)
                    )

                    if os.path.isdir(possible_row):
                        col_row_images = os.listdir(possible_row)
                        directory_structure[channel_paths[channel_idx]][col][
                            row
                        ] = col_row_images

    return directory_structure


def get_images_channel(channel_dict: dict) -> int:
    """
    Gets the number of images
    in a channel

    Parameters
    ------------
    channel_dict: dict
        Dictionary with the folder structure

    Returns
    -----------
    Number of images in the channel
    """
    n_images = 0

    for col_name, rows in channel_dict.items():
        for row_name, images in rows.items():
            len_images = len(images)

            if images == len_images:
                raise ValueError(f"Possible error in pos {col_name}/{row_name}")

            n_images += len_images

    return n_images


def get_image_metadata(image_paths: Union[PathLike, List[PathLike]]) -> List[dict]:
    """
    Function to get image metadata using
    exiftool

    Parameters
    -----------
    image_paths: Union[PathLike, List[PathLike]]
        Path(s) pointing to the images that
        we want to get the metadata from

    Returns
    -----------
    List[dict]
        List with dictionaries with the
        obtained metadata
    """

    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(image_paths)

    return metadata


def validate_rows(
    row_names: List[str],
    row_images: List[str],
    channel_path: PathLike,
    col_name: str,
    file_format: str,
    bit_depth: int,
) -> bool:
    """
    Function that validates the rows
    of a channel in a dataset. Function designed
    to work in parallel
    """

    n_images = len(row_images)

    for n_image in range(n_images):
        print(f"Validating: {col_name}/{row_names[n_image]}")
        image_paths = [
            str(
                Path(channel_path).joinpath(
                    f"{col_name}/{row_names[n_image]}/{image_path}"
                )
            )
            for image_path in row_images[n_image]
        ]

        metadata_files = get_image_metadata(image_paths)
        logger.info(f"Validating metadata: {col_name}/{row_names[n_image]}")

        for metadata_file in metadata_files:
            if metadata_file["File:FileType"] != file_format:
                msg = f"Error in file format for {metadata_file['File:SourceFile']}"
                logger.error(msg)
                return False

            elif metadata_file[f"{file_format}:BitDepth"] != bit_depth:
                msg = f"Error in bit depth for {metadata_file['File:SourceFile']}"
                logger.error(msg)
                return False

    return True


def _validate_rows(args_dict: dict) -> bool:
    """
    Function used to be dispatched to workers
    by using multiprocessing
    """
    return validate_rows(**args_dict)


def validate_metadata_parallel(
    channel_path: str, channel_dict: dict, file_format: str, bit_depth: int
) -> bool:
    """
    Validates image metadata of tiles per channel
    in parallel

    Parameters
    -----------
    channel_path: str
        Path where the channel is stored

    channel_dict: dict
        Directory structure of the channel

    file_format: str
        File format that the images have to match
        e.g., "PNG", "TIFF"

    bit_depth: int
        Bit depth that the images have to match
        e.g. 16

    Returns
    -----------
    Bool
        Boolean that indicates if the dataset
        is ready to be processed (True), or
        not (False)
    """
    workers = multiprocessing.cpu_count()
    logger.info(f"N CPUs {workers}")
    rows_per_worker = 1

    for col_name, rows in channel_dict.items():
        n_rows = len(rows)

        if n_rows < workers:
            workers = n_rows
            rows_per_worker = 1
        else:
            if n_rows % 2 != 0:
                rows_per_worker = (n_rows // workers) + 1
            else:
                rows_per_worker = n_rows // workers

        start_row = 0
        end_row = rows_per_worker
        args = []

        row_names = list(rows.keys())
        row_images = list(rows.values())

        for idx_worker in range(workers):
            arg_dict = {
                "row_names": row_names[start_row:end_row],
                "row_images": row_images[start_row:end_row],
                "channel_path": channel_path,
                "col_name": col_name,
                "file_format": file_format,
                "bit_depth": bit_depth,
            }

            args.append(arg_dict)

            if idx_worker + 1 == workers - 1:
                start_row = end_row
                end_row = n_rows
            else:
                start_row = end_row
                end_row += rows_per_worker

        res = []

        with multiprocessing.Pool(workers) as pool:
            results = pool.imap(
                _validate_rows,
                args,
                chunksize=1,
            )

            for pos in results:
                res.append(pos)

        for res_idx in range(len(res)):
            if not res[res_idx]:
                logger.error(
                    f"Dataset with format or bit depth issues found by worker {res_idx}"
                )
                return False

    return True


def validate_metadata(
    channel_path: str, channel_dict: dict, file_format: str, bit_depth: int
) -> bool:
    """
    Validates image metadata of tiles per channel
    in parallel

    Parameters
    -----------
    channel_path: str
        Path where the channel is stored

    channel_dict: dict
        Directory structure of the channel

    file_format: str
        File format that the images have to match
        e.g., "PNG", "TIFF"

    bit_depth: int
        Bit depth that the images have to match
        e.g. 16

    Returns
    -----------
    Bool
        Boolean that indicates if the dataset
        is ready to be processed (True), or
        not (False)
    """

    workers = multiprocessing.cpu_count()
    logger.info(f"N CPU cores: {workers}")

    for col_name, rows in channel_dict.items():
        for row_name, images in rows.items():
            print(f"Validating: {col_name}/{row_name}")
            start_date = datetime.now()
            image_paths = [
                str(Path(channel_path).joinpath(f"{col_name}/{row_name}/{image_path}"))
                for image_path in images
            ]

            metadata_files = get_image_metadata(image_paths)

            for metadata_file in metadata_files:
                if metadata_file["File:FileType"] != file_format:
                    msg = f"Error in file format for {metadata_file['File:SourceFile']}"
                    raise ValueError(msg)

                elif metadata_file[f"{file_format}:BitDepth"] != bit_depth:
                    msg = f"Error in bit depth for {metadata_file['File:SourceFile']}"
                    raise ValueError(msg)
            end_date = datetime.now()

            print(f"Time to validate stack of tiles: {end_date - start_date}")

    return True


def validate_dataset_metadata(dataset_path: PathLike, metadata_files: List) -> Tuple:
    """
    Validates the existance of the files
    that are in the list.

    Parameters
    ------------
    dataset_path: PathLike
        Path where the dataset is stored.
        The metadata files must be on the
        top level.

    metadata_files: List[str]
        Files to validate the existance in
        the dataset_path provided.

    Returns
    ------------
    Tuple[bool, List[str]]
        Tuple where the boolean indicates
        that all files exist in the provided
        path and a list with the missing files
        in case there is any.
    """
    dataset_path = Path(dataset_path)
    all_metadata = True
    missing_files = []

    for metadata_file in metadata_files:
        metadata_path = dataset_path.joinpath(metadata_file)

        if not metadata_path.exists():
            all_metadata = False
            missing_files.append(metadata_path.name)

    return all_metadata, missing_files


def validate_image_dataset(
    dataset_path: PathLike,
    validate_image_metadata: Optional[bool] = False,
    image_metadata_bit_depth: Optional[int] = 16,
    image_metadata_format: Optional[str] = "PNG",
) -> bool:
    """
    Validates a dataset

    Parameters
    ------------
    dataset_path: PathLike
        Path where the dataset is stored. This folder
        must have the metadata and images

    validate_image_metadata: Optional[bool]
        Validates image metadata.
        Default: True

    Returns
    -----------
    True if the dataset is correct, False otherwise
    """
    image_metadata_format = image_metadata_format.upper()
    dataset_path = Path(dataset_path)

    image_path = dataset_path.joinpath("SPIM")

    if not image_path.exists():
        # If it's not the new version, it's the old SmartSPIM
        image_path = dataset_path.joinpath("SmartSPIM")
        print("Using old version with SmartSPIM on it.")

    if not image_path.exists():
        raise FileNotFoundError(f"No SmartSPIM or SPIM paths where found")

    dataset_structure = read_image_directory_structure(image_path)

    images_per_channel = []

    for channel_name, image_paths in dataset_structure.items():
        n_images = get_images_channel(image_paths)
        images_per_channel.append(n_images)
        logger.info(f"Channel {channel_name} has {n_images} images")

    n_channels = len(images_per_channel)

    if not n_channels:
        return False

    channel_images = images_per_channel[0]

    for images_idx in range(1, n_channels):
        if channel_images != images_per_channel[images_idx]:
            return False

    if validate_image_metadata:
        logger.info("Validating image metadata")

        for channel_name, config in dataset_structure.items():
            validation_config = {
                "channel_path": str(channel_name),
                "channel_dict": config,
                "file_format": image_metadata_format,
                "bit_depth": image_metadata_bit_depth,
            }

            channel_val = validate_metadata_parallel(**validation_config)

            if not channel_val:
                logger.error(f"Wrong metadata in channel {channel_name}")
                return False

    return True
