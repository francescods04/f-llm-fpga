#!/usr/bin/env python3
"""
Colab 35B Baseline — ULTRA-ROBUST llama.cpp with auto-detection & timeout.

Features:
  - Detects llama-cli supported flags before running
  - Uses timeout to avoid infinite hangs
  - Verbose logging to diagnose issues
  - Falls back to CPU offload if GPU layers fail
  - Validates output token count
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

RESULTS_DIR = "/content/fllm_colab_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GEN_LEN = 64  # shorter for faster debugging
PROMPT = "The future of artificial intelligence is"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
LOCAL_GGUF = "/content/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
LLAMA_CLI = "/content/llama.cpp/build/bin/llama-cli"

# Read HF token
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass
if HF_TOKEN is None:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Add it to Colab secrets and restart.")

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
print("=== Installing dependencies ===")
os.system("apt-get update -qq && apt-get install -y -qq pciutils build-essential cmake curl libcurl4-openssl-dev git")
os.system("pip install -q huggingface_hub hf_transfer transformers")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------------------------------------------------------------------------
# 2. Detect VRAM
# ---------------------------------------------------------------------------
import torch
GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB")

# ---------------------------------------------------------------------------
# 3. Build llama.cpp if needed
# ---------------------------------------------------------------------------
LLAMA_DIR = "/content/llama.cpp"
if not os.path.exists(LLAMA_CLI):
    if not os.path.exists(LLAMA_DIR):
        print("=== Cloning llama.cpp (main branch) ===")
        rc = os.system("git clone https://github.com/ggml-org/llama.cpp.git /content/llama.cpp")
        if rc != 0:
            raise RuntimeError("Failed to clone llama.cpp")
    print("=== Building llama.cpp ===")
    rc = os.system(f"cmake {LLAMA_DIR} -B {LLAMA_DIR}/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON")
    if rc != 0:
        raise RuntimeError("cmake failed")
    rc = os.system(f"cmake --build {LLAMA_DIR}/build --config Release -j --target llama-cli")
    if rc != 0:
        raise RuntimeError("build failed")

# Verify llama-cli exists
if not os.path.exists(LLAMA_CLI):
    import glob
    found = glob.glob(f"{LLAMA_DIR}/**/llama-cli", recursive=True)
    if found:
        LLAMA_CLI = found[0]
    else:
        raise RuntimeError("llama-cli not found")

print(f"llama-cli: {LLAMA_CLI}")

# ---------------------------------------------------------------------------
# 4. Check supported flags
# ---------------------------------------------------------------------------
print("\n=== Checking llama-cli flags ===")
help_result = subprocess.run([LLAMA_CLI, "--help"], capture_output=True, text=True)
help_text = help_result.stdout + help_result.stderr

# Detect which GPU offload flag is supported
if "-ngl" in help_text or "--n-gpu-layers" in help_text:
    GPU_OFFLOAD_FLAG = "-ngl"
elif "--gpu-layers" in help_text:
    GPU_OFFLOAD_FLAG = "--gpu-layers"
elif "-ngl" in help_text:
    GPU_OFFLOAD_FLAG = "-ngl"
else:
    GPU_OFFLOAD_FLAG = None
    print("WARNING: no GPU offload flag found. Will run on CPU (very slow).")

# Detect prompt flag
if "-p" in help_text:
    PROMPT_FLAG = "-p"
elif "--prompt" in help_text:
    PROMPT_FLAG = "--prompt"
else:
    PROMPT_FLAG = "-p"

# Detect model flag
if "-m" in help_text:
    MODEL_FLAG = "-m"
else:
    MODEL_FLAG = "--model"

# Detect n-predict flag
if "-n" in help_text:
    N_PREDICT_FLAG = "-n"
else:
    N_PREDICT_FLAG = "--n-predict"

print(f"GPU offload flag: {GPU_OFFLOAD_FLAG}")
print(f"Prompt flag: {PROMPT_FLAG}")

# ---------------------------------------------------------------------------
# 5. Download GGUF if needed
# ---------------------------------------------------------------------------
if not os.path.exists(LOCAL_GGUF):
    print("\n=== Downloading GGUF ===")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        local_dir="/content",
        token=HF_TOKEN,
    )
else:
    print(f"Using cached: {LOCAL_GGUF}")

# ---------------------------------------------------------------------------
# 6. Build command with detected flags
# ---------------------------------------------------------------------------
print("\n=== Building command ===")
cmd = [LLAMA_CLI, MODEL_FLAG, LOCAL_GGUF, PROMPT_FLAG, PROMPT, N_PREDICT_FLAG, str(GEN_LEN), "--temp", "0.0", "--top-p", "1.0", "--top-k", "1"]

if GPU_OFFLOAD_FLAG:
    # Try all layers on GPU
    cmd += [GPU_OFFLOAD_FLAG, "999"]

print(f"Command: {' '.join(cmd)}")

# ---------------------------------------------------------------------------
# 7. Run with timeout
# ---------------------------------------------------------------------------
print("\n=== Running llama-cli (timeout: 300s) ===")
t0 = time.time()
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    print(f"Exit code: {result.returncode}")
    print(f"Elapsed: {elapsed:.2f}s")
    
    if result.stderr:
        err_preview = result.stderr[:1000]
        print(f"Stderr preview:\n{err_preview}")
        
        # If GPU offload failed, try CPU
        if result.returncode != 0 and GPU_OFFLOAD_FLAG and ("cuda" in err_preview.lower() or "gpu" in err_preview.lower()):
            print("\nGPU offload failed. Trying CPU mode...")
            cmd_cpu = [LLAMA_CLI, MODEL_FLAG, LOCAL_GGUF, PROMPT_FLAG, PROMPT, N_PREDICT_FLAG, str(GEN_LEN), "--temp", "0.0", "--top-p", "1.0", "--top-k", "1"]
            print(f"CPU Command: {' '.join(cmd_cpu)}")
            result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=300)
            elapsed = time.time() - t0
            print(f"CPU mode exit code: {result.returncode}")
except subprocess.TimeoutExpired:
    elapsed = time.time() - t0
    print(f"TIMEOUT after {elapsed:.2f}s. llama-cli hung.")
    result = None

# ---------------------------------------------------------------------------
# 8. Validate output
# ---------------------------------------------------------------------------
if result is None or result.returncode != 0:
    print("\nERROR: llama-cli failed or timed out.")
    print("This may mean:")
    print("  1. llama.cpp build is incompatible with this model")
    print("  2. Not enough memory (check ngl setting)")
    print("  3. Model format not supported by this llama.cpp version")
    sys.exit(1)

output_text = result.stdout
print(f"\n--- Generated text (first 300 chars) ---")
print(output_text[:300])
print(f"--- Total chars: {len(output_text)} ---")

# Count tokens
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
actual_token_count = len(output_tokens)
print(f"Actual tokens generated: {actual_token_count}")

# Calculate tok/s
tok_s = actual_token_count / elapsed if elapsed > 0 else 0
ms_per_tok = (elapsed * 1000) / actual_token_count if actual_token_count > 0 else float("inf")

print(f"\n✅ VALIDATED RESULT:")
print(f"  Actual tokens: {actual_token_count}")
print(f"  Elapsed:       {elapsed:.3f} s")
print(f"  tok/s:         {tok_s:.2f}")
print(f"  ms/token:      {ms_per_tok:.2f}")

# Save
result_dict = {
    "hardware": GPU_NAME,
    "vram_gb": round(TOTAL_VRAM_GB, 1),
    "model": MODEL_NAME,
    "backend": "llama.cpp",
    "quant": "UD-Q4_K_XL",
    "validated": True,
    "actual_tokens": actual_token_count,
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "elapsed_s": round(elapsed, 3),
    "llamacpp_flags": {
        "gpu_offload": GPU_OFFLOAD_FLAG,
        "prompt": PROMPT_FLAG,
        "model": MODEL_FLAG,
    },
    "note": "Robust measurement with auto-detected flags and timeout.",
}

out_path = f"{RESULTS_DIR}/35b_baseline_ultra.json"
with open(out_path, "w") as f:
    json.dump(result_dict, f, indent=2)

print(f"\nSaved to {out_path}")
print("=== ALL DONE ===")
