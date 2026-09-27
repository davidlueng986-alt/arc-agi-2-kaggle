# --- locate the model weights -------------------------------------------------
# Scans every attached Kaggle input for a HF-style snapshot instead of hardcoding
# a mount path, so the weights dataset can be swapped without editing this
# notebook. NVFP4 mirrors are rejected on sight: NVFP4 is Blackwell-native and
# L4 is Ada (sm_89).
import json
import os
from pathlib import Path

MODEL_HINTS = ("qwen3", "qwen3-8-27b", "qwen3.8", "27b")
NVFP4_MARKERS = ("nvfp4", "modelopt", "fp4")


def looks_like_hf_snapshot(directory: Path) -> bool:
    config = directory / "config.json"
    if not config.is_file():
        return False
    return any((directory / name).exists() for name in ("tokenizer_config.json", "generation_config.json")) or bool(
        list(directory.glob("*.safetensors")) or list(directory.glob("*.json.index"))
    )


def quant_of(directory: Path) -> str:
    name = str(directory).lower()
    for marker in ("nvfp4", "fp4", "awq", "int4", "gptq", "fp8", "bf16", "bnb"):
        if marker in name:
            return marker
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    text = json.dumps(config.get("quantization_config") or {}).lower()
    for marker in NVFP4_MARKERS + ("awq", "int4", "gptq", "fp8", "compressed-tensors", "bitsandbytes"):
        if marker in text:
            return "nvfp4" if marker in NVFP4_MARKERS else marker
    arch = str(config.get("architectures") or "")
    return "bf16" if "ForCausalLM" in arch or "ForConditionalGeneration" in arch else "unknown"


def size_gb(directory: Path) -> float:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            pass
    return total / (1024 ** 3)


def find_candidates():
    root = Path("/kaggle/input")
    found = []
    if not root.is_dir():
        return found
    for directory in root.rglob("config.json"):
        parent = directory.parent
        if not looks_like_hf_snapshot(parent):
            continue
        arch = ""
        try:
            arch = str(json.loads(directory.read_text(encoding="utf-8")).get("architectures") or "")
        except Exception:
            continue
        if "Qwen3" not in arch and not any(h in str(parent).lower() for h in MODEL_HINTS):
            continue
        quant = quant_of(parent)
        found.append(
            {
                "path": parent,
                "quant": quant,
                "arch": arch,
                "size_gb": round(size_gb(parent), 1),
                "blocked": quant in ("nvfp4", "fp4"),
            }
        )
    seen, unique = set(), []
    for item in found:
        resolved = str(item["path"].resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(item)
    return unique


CANDIDATES = find_candidates()
for item in CANDIDATES:
    flag = "BLOCKED(nvfp4 on Ada)" if item["blocked"] else ""
    print(f"  {item['size_gb']:>6.1f} GB  {item['quant']:<10} {item['arch'][:44]:<44} {item['path']} {flag}")

# Prefer: fp8 (25.3 GB, fits TP=4 comfortably) > bf16 (44 GB) > int4/awq.
# Anything NVFP4 is excluded because L4 is sm_89 and NVFP4 is Blackwell-native.
PREFERENCE = {"fp8": 0, "compressed-tensors": 1, "bf16": 2, "unknown": 3, "awq": 4, "int4": 5, "gptq": 6}
USABLE = [c for c in CANDIDATES if not c["blocked"]]
USABLE.sort(key=lambda c: (PREFERENCE.get(c["quant"], 9), abs(c["size_gb"] - 25.3)))

MODEL_PATH = str(USABLE[0]["path"]) if USABLE else None
MODEL_QUANT = USABLE[0]["quant"] if USABLE else None
print(f"\nselected model : {MODEL_PATH}")
print(f"quantisation   : {MODEL_QUANT}")
if MODEL_PATH is None:
    print("WARNING: no usable Qwen3 weights found among the attached inputs.")
else:
    expected_gb = 25.3 if MODEL_QUANT == "fp8" else None
    if expected_gb and abs(USABLE[0]["size_gb"] - expected_gb) > 8:
        print(f"NOTE: expected ~{expected_gb} GB for fp8, found {USABLE[0]['size_gb']} GB")
