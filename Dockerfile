# Minimalny obraz do weryfikacji, ze build na RunPod Serverless w ogole przechodzi.
# Zadnego CUDA, zadnego ACE-Step, zadnego pre-downloadu modeli.
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py .

CMD ["python", "-u", "handler.py"]
