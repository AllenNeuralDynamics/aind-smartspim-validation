# aind-smartspim-validation

This repository hosts the data validations scripts necessary to check how healthy a SmartSPIM dataset is after the uploading and before performing image processing steps in the cloud.

Some of the validations are:

- Metadata files defined in the aind-data-schema package, the following files must exist:
    - acquisition.json
    - instrument.json
    - data_description.json
    - subject.json

- Image metadata:
    - Bit depth: 16
    - Image format: PNG

- Folder structure