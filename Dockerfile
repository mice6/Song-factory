FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Instalacja zale¿noœci systemowych
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Instalacja uv (mened¿er pakietów u¿ywany przez ACE-Step)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Klonowanie ACE-Step 1.5
RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git .

# Instalacja zale¿noœci Python
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir transformers accelerate diffusers librosa soundfile numpy scipy runpod

# Pre-download modeli (przyspiesza cold start)
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; print('Pre-downloading models...')" 2>/dev/null || true

COPY handler.py .

CMD ["python3", "handler.py"]