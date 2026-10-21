FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    cuda-nvvm-12-5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3.10 -m pip install --no-cache-dir \
    tensorflow==2.21.0 \
    keras==3.12.4 \
    numpy==2.2.6 \
    pillow==12.3.0 \
    scikit-learn==1.7.2 \
    h5py==3.14.0 \
    protobuf==7.36.2

CMD ["python3.10", "--version"]
