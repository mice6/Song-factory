"""Handler RunPod Serverless - minimalny + diagnostyka srodowiska.

Zwraca to samo co wczesniej ("Hello, <name>!"), a dodatkowo raport o tym,
co faktycznie jest w uruchomionym obrazie: wersja Pythona, znacznik kroku,
SHA commita, widocznosc GPU i stan torcha. Dzieki temu sama odpowiedz mowi,
ktory obraz wstal - bez zagladania w UI RunPoda.

Diagnostyke mozna wylaczyc: {"input": {"name": "Kamil", "diagnostics": false}}
"""
import os
import platform
import subprocess
import sys

import runpod


def _gpu_kernel_test(torch):
    """Faktyczne uruchomienie kernela na karcie.

    To jedyny wiarygodny dowod, ze kola torcha zawieraja kod dla architektury
    tej karty. Sam cuda_available=True go nie daje: torch potrafi zainicjowac
    CUDA, a dopiero konkretna operacja konczy sie bledem
    "no kernel image is available for execution on the device".
    """
    try:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        wynik = float((a @ b).sum().item())
        return {"ok": True, "suma_matmul": wynik}
    except Exception as exc:
        return {"ok": False, "blad": f"{type(exc).__name__}: {exc}"}


def _torch_info():
    """Stan torcha. Do kroku 3 torcha nie ma i to jest oczekiwane."""
    try:
        import torch
    except ImportError:
        return {"installed": False}

    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            # Uwaga: get_arch_list() zwraca [] gdy is_available() jest False,
            # wiec na maszynie bez GPU ta lista nic nie mowi.
            info["arch_list"] = torch.cuda.get_arch_list()
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = "%d.%d" % torch.cuda.get_device_capability(0)
            info["kernel_test"] = _gpu_kernel_test(torch)
    except Exception as exc:
        info["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _nvidia_smi():
    """Czy worker faktycznie widzi GPU. Na runnerze GitHuba zwroci blad."""
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"niedostepne: {type(exc).__name__}"
    if proc.returncode != 0:
        return f"blad rc={proc.returncode}: {proc.stderr.strip()[:200]}"
    return proc.stdout.strip()


def _acestep_info():
    """Czy ACE-Step da sie zaimportowac w tym obrazie."""
    try:
        import acestep
    except ImportError as exc:
        return {"installed": False, "blad": f"{type(exc).__name__}: {exc}"}

    info = {"installed": True, "path": getattr(acestep, "__file__", None)}
    try:
        from importlib.metadata import version
        info["version"] = version("ace-step")
    except Exception as exc:
        info["version_blad"] = f"{type(exc).__name__}: {exc}"

    # Wlasciwe API siedzi w acestep.inference, nie w acestep_v15_pipeline
    # (ten drugi to launcher Gradio). Sam "import acestep" niczego nie dowodzi,
    # bo __init__.py zawiera tylko docstring.
    try:
        from acestep.inference import generate_music, GenerationParams
        info["inference_api"] = {
            "ok": True,
            "generate_music": callable(generate_music),
            "GenerationParams": GenerationParams.__name__,
        }
    except Exception as exc:
        info["inference_api"] = {"ok": False, "blad": f"{type(exc).__name__}: {exc}"}
    return info


def _wagi_info():
    """Czy wagi leza tam, gdzie ACE-Step ich szuka.

    ensure_model_downloaded() sprawdza {project_root}/checkpoints/<nazwa> i
    pobiera 9,4 GB, jesli katalog nie istnieje lub jest pusty. Cache HF nie
    jest w tej sciezce uzywany, bo pobieranie idzie przez local_dir=.
    """
    root = os.path.join(
        os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/ace-step"), "checkpoints")
    wymagane = ("acestep-v15-turbo", "acestep-5Hz-lm-1.7B",
                "Qwen3-Embedding-0.6B", "vae")

    if not os.path.isdir(root):
        return {"checkpoints": root, "istnieje": False}

    obecne, brakujace = [], []
    for nazwa in wymagane:
        sciezka = os.path.join(root, nazwa)
        if os.path.isdir(sciezka) and os.listdir(sciezka):
            obecne.append(nazwa)
        else:
            brakujace.append(nazwa)

    rozmiar = 0
    for katalog, _, nazwy in os.walk(root):
        for n in nazwy:
            f = os.path.join(katalog, n)
            if os.path.islink(f):
                continue
            try:
                rozmiar += os.path.getsize(f)
            except OSError:
                pass

    return {
        "checkpoints": root,
        "istnieje": True,
        "rozmiar_gb": round(rozmiar / 1024 ** 3, 2),
        "obecne": obecne,
        "brakujace": brakujace,
        "pobieranie_w_runtime": bool(brakujace),
    }


def environment():
    return {
        "step": os.environ.get("BUILD_STEP", "nieznany"),
        "git_sha": os.environ.get("GIT_SHA", "nieznany"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_version_env": os.environ.get("CUDA_VERSION"),
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_info(),
        "acestep": _acestep_info(),
        "wagi": _wagi_info(),
    }


def handler(job):
    job_input = job.get("input", {})
    name = job_input.get("name", "world")

    result = {"message": f"Hello, {name}!"}
    if job_input.get("diagnostics", True):
        result["environment"] = environment()
    return result


if __name__ == "__main__":
    env = environment()
    print(f"[startup] step={env['step']} sha={env['git_sha'][:12]}")
    print(f"[startup] python={env['python']}")
    print(f"[startup] nvidia-smi: {env['nvidia_smi']}")
    print(f"[startup] torch: {env['torch']}")
    print(f"[startup] acestep: {env['acestep']}")
    runpod.serverless.start({"handler": handler})
