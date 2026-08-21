FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip curl git wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir transformers accelerate diffusers librosa soundfile numpy runpod

COPY handler.py .

CMD ["python3", "handler.py"]
