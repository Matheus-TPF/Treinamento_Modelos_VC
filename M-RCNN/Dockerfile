FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    pycocotools opencv-python-headless scikit-learn pillow

COPY src/ /workspace/src/

CMD ["/bin/bash"]
