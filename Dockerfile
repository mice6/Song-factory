# Krok 1 odbudowy: slim + zaleznosci systemowe (git, curl, ffmpeg, libsndfile1).
# Cel: sprawdzic, czy to warstwa apt-get wiesza build na RunPod.
# Nadal zero CUDA, zero ACE-Step, zero torcha, zero pre-downloadu modeli.
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Slad w logu builda - jesli to widac, warstwa apt-get przeszla w calosci.
RUN git --version && curl --version | head -1 \
    && ffmpeg -version | head -1 \
    && ldconfig -p | grep libsndfile

RUN pip install --no-cache-dir runpod

COPY handler.py .

CMD ["python", "-u", "handler.py"]
