FROM nvcr.io/nvidia/tensorflow:22.12-tf1-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/Mask_RCNN

WORKDIR /workspace/Mask_RCNN

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "keras==2.1.6" \
    "h5py<3.0.0" \
    "numpy<1.24" \
    scipy Pillow cython matplotlib scikit-image \
    opencv-python-headless imgaug pycocotools

COPY Mask_RCNN/ /workspace/Mask_RCNN/

CMD ["/bin/bash"]
