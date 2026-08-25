# Krok 3c: baza cudnn-devel -> runtime. Bez zmian funkcjonalnych.
#
# Powod: Dockerfile z ace-step/ACE-Step-1.5 uzywa
# nvidia/cuda:12.8.1-runtime-ubuntu22.04 - czyli ani nvcc, ani naglowkow,
# ani cuDNN z bazy nie potrzeba. cuDNN przywozi torch przez pip
# (nvidia-cudnn-cu12), wiec w cudnn-devel siedzialo to podwojnie.
# Skoro upstream buduje na runtime, nic z zaleznosci nie kompiluje kerneli.
#
# Dlaczego nie ubuntu22.04 i torch 2.9.1 jak w kroku 3:
# pyproject.toml z ace-step/ACE-Step-1.5 deklaruje
#   requires-python = ">=3.11,<3.13"
#   torch==2.10.0+cu128 (linux x86_64)
# Ubuntu 22.04 daje Pythona 3.10, czyli ponizej progu. Ubuntu 24.04 daje 3.12.
#
# CUDA 12.8, bo karta na workerze to Blackwell (compute capability 12.0).
# Potwierdzone wykonaniem: arch_list zawiera sm_120, a matmul na GPU przechodzi.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Bez tego apt potrafi zawisnac na interaktywnym pytaniu tzdata o strefe czasowa.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Pierwszy slad w logu - jesli tego nie widac, build nie wystartowal.
RUN echo "===== BUILD STAMP: krok-3c / baza runtime zamiast cudnn-devel ====="

# Na 24.04 python3 to 3.12. python3-venv, bo instalujemy do wlasnego venva.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 blokuje instalacje pipem do systemowego Pythona (PEP 668,
# "externally-managed-environment"). Wlasny venv omija to czysto, bez
# --break-system-packages, i izoluje zaleznosci od pakietow dystrybucji.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# Drugi slad - warstwa systemowa i toolchain CUDA.
RUN python --version \
    && pip --version \
    && git --version \
    && ffmpeg -version | head -1 \
    && ldconfig -p | grep libsndfile \
    && (nvcc --version | tail -2 || echo "nvcc: brak w PATH (oczekiwane na bazie runtime)")

RUN pip install --no-cache-dir runpod

# Wersje wprost z pyproject.toml ACE-Step dla linux/x86_64. torchvision jest
# tam wymagany, wiec instalujemy go od razu - inaczej doszedlby w kroku 4
# i pociagnal za soba przeliczenie calej reszty.
RUN pip install --no-cache-dir \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Trzeci slad. is_available() bedzie False, bo runner GitHuba nie ma GPU,
# a get_arch_list() zwraca wtedy pusta liste - dlatego nie ma sensu jej tu
# drukowac. Prawdziwy test architektury robi handler na workerze.
RUN python -c "import torch, torchvision, torchaudio; \
print('torch', torch.__version__); \
print('torchvision', torchvision.__version__); \
print('torchaudio', torchaudio.__version__); \
print('cuda build', torch.version.cuda)"

# Znacznik kroku i SHA commita wstrzykiwane do obrazu - handler zwraca je
# w odpowiedzi, wiec od razu widac, ktory build faktycznie wstal na workerze.
# Celowo na koncu pliku: ARG zmieniajacy sie przy kazdym commicie uniewaznilby
# cache wszystkich warstw ponizej.
ARG GIT_SHA=nieznany
ENV GIT_SHA=$GIT_SHA
ENV BUILD_STEP="krok-3c: baza runtime, Python 3.12, torch 2.10.0 cu128"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
