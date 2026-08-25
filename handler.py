"""RunPod Serverless handler dla ACE-Step 1.5.

Dwa tryby, rozpoznawane po ksztalcie wejscia:

  {"input": {"caption": "...", "lyrics": "..."}}  -> generowanie muzyki
  {"input": {"name": "Kamil"}}                    -> "Hello" + diagnostyka

Tryb drugi zostaje celowo. Przez cala odbudowe sluzyl do weryfikacji obrazu
i nadal jest jedynym sposobem sprawdzenia srodowiska bez ladowania modeli.
"""
import base64
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time

import runpod

PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/ace-step")
CHECKPOINTS = os.path.join(PROJECT_ROOT, "checkpoints")
DIT_MODEL = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")

# Modele ladowane raz na worker, nie na zadanie. Ladowanie jest leniwe, zeby
# blad inicjalizacji nie ubijal kontenera przy starcie - handler ma wtedy
# zwrocic czytelny komunikat zamiast wpasc w petle restartow.
_MODELE = {"dit": None, "llm": None, "blad": None, "czas_ladowania_s": None}
_LOCK = threading.Lock()


def _zaladuj_modele():
    """Jednorazowa inicjalizacja DiT + LM. Odporna na rownolegle zadania."""
    if _MODELE["dit"] is not None or _MODELE["blad"] is not None:
        return _MODELE

    with _LOCK:
        if _MODELE["dit"] is not None or _MODELE["blad"] is not None:
            return _MODELE

        start = time.time()
        try:
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler

            dit = AceStepHandler()
            status, ok = dit.initialize_service(
                project_root=PROJECT_ROOT,
                config_path=DIT_MODEL,
                device="auto",
                use_flash_attention=True,
                compile_model=False,
                offload_to_cpu=False,
                offload_dit_to_cpu=False,
            )
            if not ok:
                raise RuntimeError("initialize_service zwrocilo ok=False: %s" % status)

            llm = LLMHandler()
            status, ok = llm.initialize(
                checkpoint_dir=CHECKPOINTS,
                lm_model_path=LM_MODEL,
                backend=os.environ.get("ACESTEP_LLM_BACKEND", "pt"),
                device="auto",
                offload_to_cpu=False,
                dtype=None,
            )
            if not ok:
                raise RuntimeError("LLMHandler.initialize zwrocilo ok=False: %s" % status)

            _MODELE["dit"], _MODELE["llm"] = dit, llm
        except Exception as exc:
            _MODELE["blad"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            _MODELE["czas_ladowania_s"] = round(time.time() - start, 1)

    return _MODELE


def _sciezka_audio(wpis):
    """Wyluskuje sciezke pliku z wpisu GenerationResult.audios.

    Dokumentacja mowi tylko "audio dictionaries with paths, keys, params",
    bez podania nazwy klucza - stad sprawdzanie kilku wariantow i fallback
    na dowolna wartosc, ktora okazuje sie istniejacym plikiem.
    """
    if isinstance(wpis, str):
        return wpis if os.path.isfile(wpis) else None
    if not isinstance(wpis, dict):
        return None
    for klucz in ("path", "audio_path", "filepath", "file", "filename", "audio"):
        wartosc = wpis.get(klucz)
        if isinstance(wartosc, str) and os.path.isfile(wartosc):
            return wartosc
    for wartosc in wpis.values():
        if isinstance(wartosc, str) and os.path.isfile(wartosc):
            return wartosc
    return None


def _generuj(job_input):
    modele = _zaladuj_modele()
    if modele["blad"]:
        return {
            "error": "Nie udalo sie zaladowac modeli",
            "szczegoly": modele["blad"],
            "czas_ladowania_s": modele["czas_ladowania_s"],
        }

    from acestep.inference import generate_music, GenerationConfig, GenerationParams

    # style_prompt to nazwa uzywana przez dotychczasowego klienta z main,
    # caption to nazwa w API ACE-Step. Przyjmujemy obie.
    caption = (job_input.get("caption")
               or job_input.get("style_prompt")
               or job_input.get("prompt")
               or "")

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=job_input.get("lyrics", ""),
        instrumental=bool(job_input.get("instrumental", False)),
        vocal_language=job_input.get("language",
                                     job_input.get("vocal_language", "unknown")),
        duration=float(job_input.get("duration", -1)),
        # 8 krokow to wartosc dla wariantu turbo, ktory mamy w obrazie.
        inference_steps=int(job_input.get("inference_steps", 8)),
        seed=int(job_input.get("seed", -1)),
        # Wnioskowanie modelu jezykowego (Chain-of-Thought): model sam dobiera
        # metadane utworu i kody semantyczne. Wszystkie przyklady w repo
        # ACE-Step maja to wlaczone, wiec domyslnie true.
        thinking=bool(job_input.get("thinking", True)),
        bpm=job_input.get("bpm"),
        keyscale=job_input.get("keyscale", ""),
        timesignature=job_input.get("timesignature", ""),
    )

    # mp3 zamiast domyslnego flac: odpowiedz wraca jako base64, a flac przy
    # 90 sekundach dalby kilkanascie MB i otarl sie o limity payloadu RunPoda.
    format_audio = job_input.get("audio_format", "mp3")
    config = GenerationConfig(
        batch_size=int(job_input.get("batch_size", 1)),
        audio_format=format_audio,
        use_random_seed=params.seed < 0,
        seeds=None if params.seed < 0 else [params.seed],
    )

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="acestep-") as katalog:
        wynik = generate_music(
            modele["dit"], modele["llm"], params, config, save_dir=katalog
        )

        if not getattr(wynik, "success", False):
            return {
                "error": "Generowanie nie powiodlo sie",
                "szczegoly": getattr(wynik, "error", None),
                "status_message": getattr(wynik, "status_message", None),
            }

        utwory = []
        for wpis in getattr(wynik, "audios", []):
            sciezka = _sciezka_audio(wpis)
            if not sciezka:
                continue
            with open(sciezka, "rb") as f:
                dane = f.read()
            wpis_slownik = wpis if isinstance(wpis, dict) else {}
            utwory.append({
                "audio_base64": base64.b64encode(dane).decode("ascii"),
                "format": os.path.splitext(sciezka)[1].lstrip(".") or format_audio,
                "rozmiar_bajtow": len(dane),
                # Seed wrocil jako null przy pierwszym udanym generowaniu, wiec
                # ACE-Step trzyma go pod innym kluczem niz zakladalem. Zanim
                # zgadne, ktorym - niech odpowiedz sama pokaze, co jest dostepne.
                "seed": wpis_slownik.get("seed", params.seed if params.seed >= 0 else None),
                "dostepne_klucze": sorted(wpis_slownik.keys()),
            })

    if not utwory:
        return {
            "error": "Generowanie zglosilo sukces, ale nie zwrocilo pliku audio",
            "status_message": getattr(wynik, "status_message", None),
        }

    return {
        # Pierwszy utwor takze na wierzchu - zgodnie z ksztaltem odpowiedzi,
        # ktorego uzywal handler na main (audio_base64 + format).
        "audio_base64": utwory[0]["audio_base64"],
        "format": utwory[0]["format"],
        "utwory": utwory,
        "czas_generowania_s": round(time.time() - start, 1),
        "czas_ladowania_modeli_s": modele["czas_ladowania_s"],
        "model": {"dit": DIT_MODEL, "lm": LM_MODEL},
        "caption": caption[:120],
    }


# ---------------------------------------------------------------------------
# Diagnostyka - sciezka sprawdzona w krokach 1-5, dziala bez ladowania modeli
# ---------------------------------------------------------------------------

def _gpu_kernel_test(torch):
    """Faktyczne uruchomienie kernela na karcie.

    Samo cuda_available=True nie wystarcza: torch potrafi zainicjowac CUDA,
    a dopiero konkretna operacja konczy sie bledem "no kernel image is
    available for execution on the device".
    """
    try:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        return {"ok": True, "suma_matmul": float((a @ b).sum().item())}
    except Exception as exc:
        return {"ok": False, "blad": "%s: %s" % (type(exc).__name__, exc)}


def _torch_info():
    try:
        import torch
    except ImportError:
        return {"installed": False}

    info = {"installed": True, "version": torch.__version__,
            "cuda_build": torch.version.cuda}
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["arch_list"] = torch.cuda.get_arch_list()
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = "%d.%d" % torch.cuda.get_device_capability(0)
            wolne, calosc = torch.cuda.mem_get_info()
            info["vram_wolne_gb"] = round(wolne / 1024 ** 3, 2)
            info["vram_calosc_gb"] = round(calosc / 1024 ** 3, 2)
            info["kernel_test"] = _gpu_kernel_test(torch)
    except Exception as exc:
        info["cuda_error"] = "%s: %s" % (type(exc).__name__, exc)
    return info


def _nvidia_smi():
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "niedostepne: %s" % type(exc).__name__
    if proc.returncode != 0:
        return "blad rc=%s: %s" % (proc.returncode, proc.stderr.strip()[:200])
    return proc.stdout.strip()


def _acestep_info():
    try:
        import acestep
    except ImportError as exc:
        return {"installed": False, "blad": "%s: %s" % (type(exc).__name__, exc)}

    info = {"installed": True, "path": getattr(acestep, "__file__", None)}
    try:
        from importlib.metadata import version
        info["version"] = version("ace-step")
    except Exception as exc:
        info["version_blad"] = "%s: %s" % (type(exc).__name__, exc)

    try:
        from acestep.inference import generate_music, GenerationParams
        info["inference_api"] = {"ok": True,
                                 "generate_music": callable(generate_music),
                                 "GenerationParams": GenerationParams.__name__}
    except Exception as exc:
        info["inference_api"] = {"ok": False,
                                 "blad": "%s: %s" % (type(exc).__name__, exc)}

    info["modele_zaladowane"] = _MODELE["dit"] is not None
    if _MODELE["blad"]:
        info["blad_ladowania"] = _MODELE["blad"]
    return info


def _wagi_info():
    """Czy wagi leza tam, gdzie ACE-Step ich szuka.

    ensure_model_downloaded() sprawdza {project_root}/checkpoints/<nazwa>
    i pobiera 9,4 GB, jesli katalog nie istnieje albo jest pusty.
    """
    wymagane = ("acestep-v15-turbo", "acestep-5Hz-lm-1.7B",
                "Qwen3-Embedding-0.6B", "vae")
    if not os.path.isdir(CHECKPOINTS):
        return {"checkpoints": CHECKPOINTS, "istnieje": False}

    obecne, brakujace = [], []
    for nazwa in wymagane:
        sciezka = os.path.join(CHECKPOINTS, nazwa)
        if os.path.isdir(sciezka) and os.listdir(sciezka):
            obecne.append(nazwa)
        else:
            brakujace.append(nazwa)

    rozmiar = 0
    for katalog, _, nazwy in os.walk(CHECKPOINTS):
        for n in nazwy:
            f = os.path.join(katalog, n)
            if os.path.islink(f):
                continue
            try:
                rozmiar += os.path.getsize(f)
            except OSError:
                pass

    return {
        "checkpoints": CHECKPOINTS,
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

    generuj = any(job_input.get(k) for k in
                  ("caption", "style_prompt", "prompt", "lyrics", "instrumental"))
    if generuj:
        return _generuj(job_input)

    wynik = {"message": "Hello, %s!" % job_input.get("name", "world")}
    if job_input.get("diagnostics", True):
        wynik["environment"] = environment()
    return wynik


if __name__ == "__main__":
    env = environment()
    print("[startup] step=%s sha=%s" % (env["step"], env["git_sha"][:12]))
    print("[startup] python=%s" % env["python"])
    print("[startup] nvidia-smi: %s" % env["nvidia_smi"])
    print("[startup] wagi: %s" % env["wagi"])
    print("[startup] modele laduja sie leniwie, przy pierwszym zadaniu generowania")
    runpod.serverless.start({"handler": handler})
