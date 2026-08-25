# Krok 6: realny handler generujacy audio. Dockerfile bez zmian poza
# znacznikiem - caly ciezar tego kroku jest w handler.py.
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
RUN echo "===== BUILD STAMP: krok-6 / handler generujacy audio ====="

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
# Wagi musza lezec w {project_root}/checkpoints, a NIE w cache'u HF.
# acestep/api/model_download.py:ensure_model_downloaded robi:
#     model_path = os.path.join(checkpoint_dir, model_name)
#     if os.path.exists(model_path) and os.listdir(model_path): return model_path
# czyli pomija pobieranie wylacznie wtedy, gdy podkatalog juz istnieje.
# Pobieranie idzie przez snapshot_download(local_dir=...), ktore omija cache HF -
# wagi wpieczone w HF_HOME (krok 5) nie zostalyby uzyte i worker sciagalby
# 9,4 GB przy pierwszym zadaniu.
#
# Repo ACE-Step/Ace-Step1.5 jest "unified": rozpakowuje sie prosto do
# checkpoints/ i tworzy tam wszystkie cztery podkatalogi naraz.
ENV ACESTEP_PROJECT_ROOT=/opt/ace-step
ENV HF_HOME=/opt/hf-cache

# Bez tego ensure_model_downloaded probuje w runtime polaczyc sie z
# www.google.com:443 (can_access_google), zeby wybrac zrodlo. Przy komplecie
# wag ta sciezka i tak nie powinna sie wykonac, ale nie chcemy sondy sieciowej
# w cold starcie workera.
ENV ACESTEP_DOWNLOAD_SOURCE=huggingface

ARG ACESTEP_HF_REPO=ACE-Step/Ace-Step1.5
RUN python -c "from huggingface_hub import snapshot_download; p = snapshot_download('$ACESTEP_HF_REPO', local_dir='/opt/ace-step/checkpoints'); print('wagi pobrane do:', p)"

# Twardy warunek powodzenia. Sprawdza dokladnie to, co sprawdza ACE-Step:
# istnienie i niepustosc kazdego podkatalogu w checkpoints/.
# Liczy tylko pliki rzeczywiste (bez dowiazan), wiec rozmiar nie jest zawyzony.
RUN python -c "import os; root='/opt/ace-step/checkpoints'; wymagane=['acestep-v15-turbo','acestep-5Hz-lm-1.7B','Qwen3-Embedding-0.6B','vae']; brak=[k for k in wymagane if not (os.path.isdir(os.path.join(root,k)) and os.listdir(os.path.join(root,k)))]; assert not brak, 'brak lub pusty katalog: %s' % brak; rozmiar=sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(root) for f in fs if not os.path.islink(os.path.join(d,f))); print('checkpoints: %.2f GB' % (rozmiar/1024**3)); assert rozmiar > 8*1024**3, 'wagi za male, pobranie niepelne'; print('wagi OK - ACE-Step pominie pobieranie w runtime')"

# Znacznik kroku i SHA commita wstrzykiwane do obrazu - handler zwraca je
# w odpowiedzi, wiec od razu widac, ktory build faktycznie wstal na workerze.
# Celowo na koncu pliku: ARG zmieniajacy sie przy kazdym commicie uniewaznilby
# cache wszystkich warstw ponizej.
ARG GIT_SHA=nieznany
ENV GIT_SHA=$GIT_SHA
ENV BUILD_STEP="krok-6: handler generujacy audio (ACE-Step turbo + LM 1.7B)"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
