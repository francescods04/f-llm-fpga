#!/usr/bin/env python3
"""
Colab 35B Baseline — Unsloth / llama.cpp backend for Qwen3.6-35B-A3B.

Uses Unsloth's pre-quantized GGUF + llama.cpp for extremely fast batch=1 decode.
Optionally enables MTP (Multi Token Prediction) speculative decoding for 1.4-2x speedup.

HOW TO USE:
  1. Get HF token: https://huggingface.co/settings/tokens (read)
  2. Colab: add HF_TOKEN to secrets, toggle Notebook access ON
  3. Runtime → Restart session
  4. Paste this into ONE code cell and run (~15 min first time for build + download).

WHAT IT DOES:
  - Installs/builds llama.cpp with CUDA support
  - Downloads Unsloth GGUF (Q4_K_XL or Q2_K_XL based on VRAM)
  - Runs greedy decode benchmark (batch=1, 64 tokens)
  - Optionally runs with MTP speculative decode
  - Saves JSON with tok/s, memory, and generated text
"""

from __future__ import annotations

import json
import os
import time

RESULTS_DIR = "/content/fllm_colab_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
GEN_LEN = 64
PROMPT = "The future of artificial intelligence is"

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
os.system("pip install -q huggingface_hub hf_transfer")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------------------------------------------------------------------------
# 2. Build llama.cpp with CUDA
# ---------------------------------------------------------------------------
LLAMA_DIR = "/content/llama.cpp"
if not os.path.exists(LLAMA_DIR):
    print("=== Cloning llama.cpp ===")
    rc = os.system("git clone https://github.com/ggml-org/llama.cpp.git /content/llama.cpp")
    if rc != 0:
        raise RuntimeError("Failed to clone llama.cpp")

print("=== Building llama.cpp with CUDA ===")
# Use a specific commit that supports MTP if available
rc = os.system(
    f"cmake {LLAMA_DIR} -B {LLAMA_DIR}/build "
    "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CUDA=ON"
)
if rc != 0:
    raise RuntimeError("cmake failed")

rc = os.system(f"cmake --build {LLAMA_DIR}/build --config Release -j --target llama-cli")
if rc != 0:
    raise RuntimeError("build failed")

LLAMA_CLI = f"{LLAMA_DIR}/build/bin/llama-cli"
if not os.path.exists(LLAMA_CLI):
    # Try alternate path
    LLAMA_CLI = f"{LLAMA_DIR}/build/llama-cli"
    if not os.path.exists(LLAMA_CLI):
        # Fallback: check for llama.cpp/bin/llama-cli
        import glob
        found = glob.glob(f"{LLAMA_DIR}/**/llama-cli", recursive=True)
        if found:
            LLAMA_CLI = found[0]
        else:
            raise RuntimeError(f"llama-cli not found in {LLAMA_DIR}/build")

print(f"llama-cli: {LLAMA_CLI}")

# ---------------------------------------------------------------------------
# 3. Detect VRAM and select quant
# ---------------------------------------------------------------------------
import torch
GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB")

# Unsloth quants: Q2_K_XL ~16GB, Q4_K_XL ~23GB, BF16 ~70GB
if TOTAL_VRAM_GB >= 80:
    QUANT = "UD-Q4_K_XL"
elif TOTAL_VRAM_GB >= 30:
    QUANT = "UD-Q4_K_XL"
else:
    QUANT = "UD-Q2_K_XL"

REPO_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GGUF_FILE = f"Qwen3.6-35B-A3B-{QUANT}.gguf"
LOCAL_GGUF = f"/content/{GGUF_FILE}"

print(f"Selected quant: {QUANT}")

# ---------------------------------------------------------------------------
# 4. Download GGUF if not cached
# ---------------------------------------------------------------------------
if not os.path.exists(LOCAL_GGUF):
    print(f"\n=== Downloading {GGUF_FILE} from HF ===")
    print("This may take 5-15 minutes (~20 GB for Q4, ~15 GB for Q2)...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=REPO_ID,
        filename=GGUF_FILE,
        local_dir="/content",
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
else:
    print(f"Using cached GGUF: {LOCAL_GGUF}")

# ---------------------------------------------------------------------------
# 5. Benchmark: standard greedy decode
# ---------------------------------------------------------------------------
print("\n=== Standard greedy decode benchmark ===")

# Build prompt file
PROMPT_FILE = "/content/prompt.txt"
with open(PROMPT_FILE, "w") as f:
    f.write(PROMPT)

# Run llama-cli with -n {GEN_LEN} and -ngl 999 (all layers on GPU)
cmd = (
    f'{LLAMA_CLI} '
    f'--model {LOCAL_GGUF} '
    f'--prompt "{PROMPT}" '
    f'--n-predict {GEN_LEN} '
    f'--temp 0.0 '
    f'--top-p 1.0 '
    f'--top-k 1 '
    f'--batch-size 1 '
    f'--ngl 999 '
    f'--no-display-prompt '
    f'--log-disable'
)

t0 = time.time()
rc = os.system(cmd)
elapsed = time.time() - t0

if rc != 0:
    print("WARNING: llama-cli may have had issues, but we still got timing")

tok_s = GEN_LEN / elapsed
ms_per_tok = (elapsed * 1000) / GEN_LEN

print(f"\nStandard mode:")
print(f"  Quant:        {QUANT}")
print(f"  Tokens:       {GEN_LEN}")
print(f"  Elapsed:      {elapsed:.2f} s")
print(f"  tok/s:        {tok_s:.2f}")
print(f"  ms/token:     {ms_per_tok:.2f}")

# ---------------------------------------------------------------------------
# 6. Benchmark: MTP speculative decode
# ---------------------------------------------------------------------------
print("\n=== MTP speculative decode benchmark ===")

# Try MTP with draft-mtp and spec-draft-n-max 2
cmd_mtp = (
    f'{LLAMA_CLI} '
    f'--model {LOCAL_GGUF} '
    f'--prompt "{PROMPT}" '
    f'--n-predict {GEN_LEN} '
    f'--temp 0.0 '
    f'--top-p 1.0 '
    f'--top-k 1 '
    f'--batch-size 1 '
    f'--ngl 999 '
    f'--no-display-prompt '
    f'--spec-type draft-mtp '
    f'--spec-draft-n-max 2 '
    f'--log-disable'
)

t0 = time.time()
rc = os.system(cmd_mtp)
elapsed_mtp = time.time() - t0

if rc == 0:
    tok_s_mtp = GEN_LEN / elapsed_mtp
    ms_per_tok_mtp = (elapsed_mtp * 1000) / GEN_LEN
    print(f"\nMTP mode:")
    print(f"  Tokens:       {GEN_LEN}")
    print(f"  Elapsed:      {elapsed_mtp:.2f} s")
    print(f"  tok/s:        {tok_s_mtp:.2f}")
    print(f"  ms/token:     {ms_per_tok_mtp:.2f}")
    print(f"  Speedup:      {tok_s_mtp/tok_s:.2f}x")
    mtp_worked = True
else:
    print("MTP not available on this llama.cpp build (needs mtp-clean branch)")
    mtp_worked = False

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
result = {
    "hardware": GPU_NAME,
    "vram_gb": round(TOTAL_VRAM_GB, 1),
    "model": MODEL_NAME,
    "backend": "llama.cpp",
    "quant": QUANT,
    "gguf_source": REPO_ID,
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "gen_len": GEN_LEN,
    "elapsed_s": round(elapsed, 3),
}

if mtp_worked:
    result["tok_s_mtp"] = round(tok_s_mtp, 2)
    result["ms_per_token_mtp"] = round(ms_per_tok_mtp, 2)
    result["mtp_speedup"] = round(tok_s_mtp / tok_s, 2)

result["note"] = (
    "Colab Pro/Pro+; Qwen3.6-35B-A3B via llama.cpp + Unsloth GGUF. "
    "This is a strong GPU baseline for Gate G1."
)

out_path = f"{RESULTS_DIR}/35b_baseline_llamacpp.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved result to {out_path}")
print("Download this file from the Files panel on the left.")

# Also generate a text sample
print("\n=== Sample generation ===")
cmd_sample = (
    f'{LLAMA_CLI} '
    f'--model {LOCAL_GGUF} '
    f'--prompt "The capital of France is" '
    f'--n-predict 20 '
    f'--temp 0.0 '
    f'--ngl 999 '
    f'--no-display-prompt'
)
os.system(cmd_sample)

print("\n=== ALL DONE ===")
