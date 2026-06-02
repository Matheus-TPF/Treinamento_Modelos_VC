FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/YOLOX

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/YOLOX
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "numpy<2.0.0" opencv-python-headless

CMD [ "/bin/bash" ]