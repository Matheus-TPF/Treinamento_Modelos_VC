FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/YOLOX

RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git .

RUN pip install --no-cache-dir -U pip==23.3.2 setuptools==69.0.3 wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-build-isolation -e .

CMD [ "/bin/bash" ]