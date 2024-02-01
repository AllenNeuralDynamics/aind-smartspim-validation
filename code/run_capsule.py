""" top level run script """
import logging
import multiprocessing
import os
from datetime import datetime

import aind_smartspim_validation.utilities as utils
from aind_smartspim_validation.validate import (validate_dataset_metadata,
                                                validate_image_dataset)


def create_logger(output_log_path: str) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/validation_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger


def run():
    """basic run function"""

    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    logger = create_logger(output_log_path=results_folder)
    utils.print_system_information(logger)

    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    logger.info(f"{'='*40} SmartSPIM Validation {'='*40}")

    logger.info("Starting metadata validation")

    metadata_status, missing_files = validate_dataset_metadata(
        dataset_path=data_folder,
        metadata_files=[
            "acquisition.json",
            "instrument.json",
            "data_description.json",
            "subject.json",
            "derivatives/metadata.json",
            "derivatives/DarkMaster_cropped.tif",
            # "procedures.json"
        ],
    )

    if not metadata_status:
        utils.stop_child_process(profile_process)
        raise ValueError(f"The following metadata is missing {missing_files}.")

    logger.info("Validating image metadata")

    image_status = validate_image_dataset(
        dataset_path=data_folder,
        validate_image_metadata=False,
    )

    if not image_status:
        utils.stop_child_process(profile_process)
        raise ValueError("Error validating image dataset.")

    logger.info("Dataset image validation check -> passed")
    logger.info("Dataset metadata validation check -> passed")

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            results_folder,
            "smartspim_validation",
        )


if __name__ == "__main__":
    run()
