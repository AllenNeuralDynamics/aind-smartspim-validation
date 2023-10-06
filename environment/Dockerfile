FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN cd /home && \
    wget https://exiftool.org/Image-ExifTool-12.57.tar.gz --no-check-certificate && \
    gzip -dc Image-ExifTool-12.57.tar.gz | tar -xf -

ENV PATH "/home/Image-ExifTool-12.57:$PATH"

RUN pip install -U --no-cache-dir \
    tqdm==4.64.1 \
    PyExifTool==0.5.5