# Krok 5: pre-download wag do obrazu. Bez zmiany logiki handlera -
# ten krok konczy sie na tym, ze wagi leza w cache'u HF wewnatrz obrazu.
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
RUN echo "===== BUILD STAMP: krok-5 / pre-download wag ACE-Step ====="

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

# ---- ACE-Step 1.5 -------------------------------------------------------
# uv, tak jak w Dockerfile upstreamu. Instalacja pipem nie wchodzi w gre:
# pyproject.toml ma [tool.uv.sources] dla nano-vllm, czego pip nie rozumie.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Klon do osobnego katalogu, nie do /app. Oryginalny Dockerfile na main robil
# "git clone ... ." prosto do WORKDIR, gdzie chwile pozniej ladowal COPY
# handler.py - nasz kod mieszal sie z ich repo.
# Commit przypiety: build ma byc powtarzalny, a ruchome main bylo by kolejna
# zmienna przy diagnozowaniu "na ktorym kroku sie zepsulo".
ARG ACESTEP_COMMIT=14c0211d5a0653b0f63e27686f4c3f151b4d8629
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ace-step     && cd /opt/ace-step     && git checkout --quiet "$ACESTEP_COMMIT"     && git --no-pager log -1 --format="ACE-Step przypiety na %h z %ad" --date=short

# uv instaluje do NASZEGO venva zamiast tworzyc wlasny - dzieki temu widzi juz
# zainstalowanego torcha 2.10.0+cu128 jako spelnionego i nie ciagnie go drugi raz.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /opt/ace-step
RUN uv sync --frozen --no-dev
WORKDIR /app

# runpod dopiero teraz: "uv sync" usuwa z venva pakiety spoza uv.lock,
# a runpod tam nie wystepuje. Zainstalowany wczesniej zostalby skasowany.
RUN pip install --no-cache-dir runpod

# Czwarty slad - twardy warunek powodzenia kroku 4. Jesli import nie przejdzie,
# build ma pasc tutaj, w logu, a nie po cichu na workerze.
RUN python -c "import acestep; print('acestep OK:', acestep.__file__)"     && python -c "from acestep.inference import generate_music, GenerationParams; print('acestep.inference OK:', generate_music.__name__, GenerationParams.__name__)"     && python -c "import runpod; print('runpod OK')"

# ---- wagi modeli ---------------------------------------------------------
# HF_HOME wskazuje na katalog wewnatrz obrazu, wiec kazde pozniejsze
# from_pretrained / snapshot_download trafia w gotowy cache zamiast pobierac
# cokolwiek na workerze. To tez usuwa zaleznosc od dostepnosci HF w runtime.
ENV HF_HOME=/opt/hf-cache

# Pobieramy TYLKO ACE-Step/Ace-Step1.5 (9,4 GB). To kompletny zestaw roboczy:
#   acestep-v15-turbo/     4,46 GB  - DiT
#   acestep-5Hz-lm-1.7B/   3,50 GB  - model jezykowy, w komplecie
#   Qwen3-Embedding-0.6B/  1,12 GB
#   vae/                   0,31 GB
# Osobne repo acestep-5Hz-lm-4B (7,85 GB) pomijamy swiadomie: wariant 1.7B
# jest juz tutaj, a wg constants.py potrzebuje 8 GB VRAM zamiast 12 GB.
# Przy plastrze MIG 1g.24gb to wiekszy zapas i o 8 GB mniejszy obraz.
ARG ACESTEP_HF_REPO=ACE-Step/Ace-Step1.5
RUN python -c "from huggingface_hub import snapshot_download; p = snapshot_download('$ACESTEP_HF_REPO'); print('wagi pobrane do:', p)"

# Piaty slad - twardy warunek powodzenia kroku 5. Sprawdza, ze kluczowe pliki
# faktycznie sa na dysku, a nie ze samo snapshot_download zwrocilo sciezke.
RUN python -c "import os, glob; root='/opt/hf-cache'; pliki=[f for f in glob.glob(root+'/**/*', recursive=True) if os.path.isfile(f)]; rozmiar=sum(os.path.getsize(f) for f in pliki); print('plikow: %d, razem: %.2f GB' % (len(pliki), rozmiar/1024**3)); brak=[k for k in ['acestep-v15-turbo','acestep-5Hz-lm-1.7B','Qwen3-Embedding-0.6B','vae'] if not any(k in f for f in pliki)]; assert not brak, 'brakuje katalogow: %s' % brak; assert rozmiar > 8*1024**3, 'wagi za male, pobranie niepelne'; print('wagi OK')"

# Znacznik kroku i SHA commita wstrzykiwane do obrazu - handler zwraca je
# w odpowiedzi, wiec od razu widac, ktory build faktycznie wstal na workerze.
# Celowo na koncu pliku: ARG zmieniajacy sie przy kazdym commicie uniewaznilby
# cache wszystkich warstw ponizej.
ARG GIT_SHA=nieznany
ENV GIT_SHA=$GIT_SHA
ENV BUILD_STEP="krok-5: ACE-Step 1.5 + wagi wpieczone w obraz"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
