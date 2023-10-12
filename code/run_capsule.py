""" top level run script """
import logging
import os

from aind_smartspim_validation.validate import (validate_dataset_metadata,
                                                validate_image_dataset)

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


def run():
    """basic run function"""
    data_folder = os.path.abspath("../data")

    logger.info("Starting metadata validation")

    metadata_status, missing_files = validate_dataset_metadata(
        dataset_path=data_folder,
        metadata_files=[
            "acquisition.json",
            "instrument.json",
            "data_description.json",
            "subject.json",
            # "procedures.json"
        ],
    )

    if not metadata_status:
        raise ValueError(f"The following metadata is missing {missing_files}.")

    logger.info("Validating image metadata")

    image_status = validate_image_dataset(
        dataset_path=data_folder,
        validate_image_metadata=False,
    )

    if not image_status:
        raise ValueError("Error validating image dataset.")

    logger.info(f"Dataset image validation check -> passed")
    logger.info(f"Dataset metadata validation check -> passed")


if __name__ == "__main__":
    run()
