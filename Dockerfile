# Krok 2b: podbicie CUDA z 12.1.1 na 12.8.1. Nadal bez torcha i bez ACE-Step.
#
# Powod: worker RunPoda oddaje karte "RTX PRO 6000 Blackwell Server Edition".
# CUDA 12.1 nie zna architektury Blackwell (sm_120) - wsparcie weszlo w 12.8.
# Na 12.1 torch zbudowalby sie bez bledu i dopiero pierwsze wywolanie na GPU
# zwrociloby "no kernel image is available for execution on the device".
#
# W nowszych tagach nvidia/cuda "cudnn8" nazywa sie juz "cudnn".
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Bez tego apt potrafi zawisnac na interaktywnym pytaniu tzdata o strefe czasowa.
# Oryginalny Dockerfile tego nie mial.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Pierwszy slad w logu - jesli tego nie widac, build nie wystartowal.
RUN echo "===== BUILD STAMP: krok-2b / baza CUDA 12.8.1-cudnn-devel ====="

# Baza CUDA to czysta Ubuntu 22.04 - Pythona trzeba doinstalowac (python3 = 3.10).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Drugi slad - potwierdza, ze warstwa systemowa i toolchain CUDA sa na miejscu.
RUN python3 --version \
    && pip3 --version \
    && git --version \
    && ffmpeg -version | head -1 \
    && ldconfig -p | grep libsndfile \
    && (nvcc --version | tail -2 || echo "nvcc: brak w PATH")

RUN pip3 install --no-cache-dir runpod

# Znacznik kroku i SHA commita wstrzykiwane do obrazu - handler zwraca je
# w odpowiedzi, wiec od razu widac, ktory build faktycznie wstal na workerze.
# Celowo na koncu pliku: ARG zmieniajacy sie przy kazdym commicie uniewaznilby
# cache wszystkich warstw ponizej.
ARG GIT_SHA=nieznany
ENV GIT_SHA=$GIT_SHA
ENV BUILD_STEP="krok-2b: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
