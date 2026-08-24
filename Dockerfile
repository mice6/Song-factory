# Krok 3: torch + torchaudio z kanalu cu128. Nadal bez ACE-Step.
#
# Baza CUDA 12.8.1, bo worker RunPoda oddaje karte
# "RTX PRO 6000 Blackwell Server Edition" o compute_cap 12.0 (sm_120).
# CUDA 12.1 nie zna Blackwella - wsparcie weszlo dopiero w 12.8.
#
# W nowszych tagach nvidia/cuda "cudnn8" nazywa sie juz "cudnn".
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Bez tego apt potrafi zawisnac na interaktywnym pytaniu tzdata o strefe czasowa.
# Oryginalny Dockerfile tego nie mial.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Pierwszy slad w logu - jesli tego nie widac, build nie wystartowal.
RUN echo "===== BUILD STAMP: krok-3 / CUDA 12.8.1 + torch cu128 ====="

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

# Kanal cu128, nie cu121: karta na workerze to Blackwell (sm_120), a kola cu121
# nie zawieraja kernela dla tej architektury. cu128 pokrywa zakres od Ampere po
# Blackwella, wiec dziala niezaleznie od tego, co RunPod przydzieli.
# Wersje przypiete swiadomie - przy diagnozowaniu "na ktorym kroku sie zepsulo"
# ruchomy latest bylby dodatkowa zmienna.
RUN pip3 install --no-cache-dir \
        torch==2.9.1 \
        torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Trzeci slad. torch.cuda.is_available() zwroci tu False i to jest OCZEKIWANE -
# runner GitHuba nie ma GPU. Istotny jest "arch list": musi zawierac sm_120,
# inaczej karta na workerze nie zostanie obsluzona.
RUN python3 -c "import torch; \
print('torch', torch.__version__); \
print('cuda build', torch.version.cuda); \
print('arch list', torch.cuda.get_arch_list()); \
print('is_available (brak GPU na runnerze)', torch.cuda.is_available())"

# Znacznik kroku i SHA commita wstrzykiwane do obrazu - handler zwraca je
# w odpowiedzi, wiec od razu widac, ktory build faktycznie wstal na workerze.
# Celowo na koncu pliku: ARG zmieniajacy sie przy kazdym commicie uniewaznilby
# cache wszystkich warstw ponizej.
ARG GIT_SHA=nieznany
ENV GIT_SHA=$GIT_SHA
ENV BUILD_STEP="krok-3: CUDA 12.8.1 + torch 2.9.1 cu128"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
