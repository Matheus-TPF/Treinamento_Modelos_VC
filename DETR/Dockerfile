FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/detr

COPY detr/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install "numpy<1.24" cython && \
    pip install -r requirements.txt "numpy<1.24" --no-build-isolation

COPY detr/ ./

VOLUME ["/workspace/detr/coco", "/workspace/detr/outputs", "/workspace/detr/weights"]

CMD ["/bin/bash"]
