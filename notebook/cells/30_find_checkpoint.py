"""Locate the Qwen3.8-27B HF checkpoint among the attached Kaggle datasets."""
import glob
import os

MODEL_SLUG_CANDIDATES = [
    "qwen3-8-27b-fp8",
    "qwen3-8-27b-fp8-hf-snapshot",
    "qwen3-8-27b-bf16",
    "qwen3-8-27b",
]


def is_hf_checkpoint(path):
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    weights = glob.glob(os.path.join(path, "*.safetensors")) + glob.glob(os.path.join(path, "*.bin"))
    return bool(weights)


def find_checkpoint():
    tried = []
    for slug in MODEL_SLUG_CANDIDATES:
        base = os.path.join(INPUT_ROOT, slug)
        tried.append(base)
        if is_hf_checkpoint(base):
            return base
        if os.path.isdir(base):
            for sub in sorted(os.listdir(base)):
                cand = os.path.join(base, sub)
                if is_hf_checkpoint(cand):
                    return cand
    if os.path.isdir(INPUT_ROOT):
        for entry in sorted(os.listdir(INPUT_ROOT)):
            cand = os.path.join(INPUT_ROOT, entry)
            if "qwen3" in entry.lower() and "27b" in entry.lower():
                if is_hf_checkpoint(cand):
                    return cand
                if os.path.isdir(cand):
                    for sub in sorted(os.listdir(cand)):
                        deep = os.path.join(cand, sub)
                        if is_hf_checkpoint(deep):
                            return deep
    raise RuntimeError("no HF checkpoint found; looked at %s" % tried)


def checkpoint_size_gb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1e9


try:
    MODEL_PATH = find_checkpoint()
    size_gb = checkpoint_size_gb(MODEL_PATH)
    log("checkpoint: %s (%.1f GB)" % (MODEL_PATH, size_gb))
    if size_gb < 20:
        log("WARNING: %.1f GB is small for a 27B FP8/BF16 checkpoint - verify it is complete." % size_gb)
    if "gguf" in MODEL_PATH.lower() or "ollama" in MODEL_PATH.lower():
        log("ERROR: %s looks like an Ollama/GGUF archive. vLLM needs an HF checkpoint." % MODEL_PATH)
        MODEL_PATH = None
except RuntimeError as exc:
    MODEL_PATH = None
    log("checkpoint discovery failed: %s" % exc)

print("elapsed %.0fs, remaining %.2f h" % (elapsed_s(), remaining_s() / 3600.0))
