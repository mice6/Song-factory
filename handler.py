"""Minimalny handler RunPod Serverless - smoke test builda."""
import runpod


def handler(job):
    job_input = job.get("input", {})
    name = job_input.get("name", "world")
    return {"message": f"Hello, {name}!"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
