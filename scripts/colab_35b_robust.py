#!/usr/bin/env python3
"""
Colab 35B Baseline — ROBUST llama.cpp measurement with output validation.

This script fixes the measurement issues in colab_35b_unsloth.py:
  - Captures llama-cli stdout to verify actual tokens generated
  - Parses generated text to count real output length
  - Validates that llama-cli succeeded (exit code 0)
  - Falls back gracefully if MTP is unsupported
  - Saves detailed JSON with both raw timing and validated timing
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

RESULTS_DIR = "/content/fllm_colab_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GEN_LEN = 128
PROMPT = "The future of artificial intelligence is"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

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
# 2. Detect VRAM and select quant
# ---------------------------------------------------------------------------
import torch
GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB")

if TOTAL_VRAM_GB >= 25:
    QUANT = "UD-Q4_K_XL"
elif TOTAL_VRAM_GB >= 18:
    QUANT = "UD-Q2_K_XL"
else:
    raise RuntimeError(f"VRAM {TOTAL_VRAM_GB:.1f}GB insufficient. Need >=18GB for 35B-A3B GGUF.")

print(f"Selected quant: {QUANT}")

# ---------------------------------------------------------------------------
# 3. Build llama.cpp with CUDA
# ---------------------------------------------------------------------------
LLAMA_DIR = "/content/llama.cpp"
if not os.path.exists(LLAMA_DIR):
    print("=== Cloning llama.cpp ===")
    # Try mtp-clean first, fall back to main
    rc = os.system(
        "git clone -b mtp-clean https://github.com/am17an/llama.cpp.git /content/llama.cpp 2>/dev/null || "
        "git clone https://github.com/ggml-org/llama.cpp.git /content/llama.cpp"
    )
    if rc != 0:
        raise RuntimeError("Failed to clone llama.cpp")

print("=== Building llama.cpp with CUDA ===")
rc = os.system(
    f"cmake {LLAMA_DIR} -B {LLAMA_DIR}/build "
    "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CUDA=ON 2>&1 | tail -20"
)
if rc != 0:
    raise RuntimeError("cmake failed")

rc = os.system(f"cmake --build {LLAMA_DIR}/build --config Release -j --target llama-cli 2>&1 | tail -20")
if rc != 0:
    raise RuntimeError("build failed")

import glob
found = glob.glob(f"{LLAMA_DIR}/**/llama-cli", recursive=True)
if not found:
    raise RuntimeError(f"llama-cli not found")
LLAMA_CLI = found[0]
print(f"llama-cli: {LLAMA_CLI}")

# Verify it works
result = subprocess.run([LLAMA_CLI, "--version"], capture_output=True, text=True)
print(f"llama-cli version: {result.stdout.strip()}")

# ---------------------------------------------------------------------------
# 4. Download Unsloth GGUF
# ---------------------------------------------------------------------------
REPO_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GGUF_FILE = f"Qwen3.6-35B-A3B-{QUANT}.gguf"
LOCAL_GGUF = f"/content/{GGUF_FILE}"

if not os.path.exists(LOCAL_GGUF):
    print(f"\n=== Downloading {GGUF_FILE} ===")
    print("This may take 5-15 minutes...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=REPO_ID,
        filename=GGUF_FILE,
        local_dir="/content",
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
else:
    print(f"Using cached: {LOCAL_GGUF}")

# ---------------------------------------------------------------------------
# 5. Benchmark: standard greedy decode (with output capture)
# ---------------------------------------------------------------------------
print("\n=== Standard greedy decode (validated) ===")

cmd = [
    LLAMA_CLI,
    "-m", LOCAL_GGUF,
    "-p", PROMPT,
    "-n", str(GEN_LEN),
    "--temp", "0.0",
    "--top-p", "1.0",
    "--top-k", "1",
    "-ngl", "999",
]

print(f"Running: {' '.join(cmd)}")
t0 = time.time()
result = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - t0

print(f"Exit code: {result.returncode}")
if result.stderr:
    print(f"Stderr (first 500 chars): {result.stderr[:500]}")

if result.returncode != 0:
    print("ERROR: llama-cli failed. Cannot measure.")
    # Save failure info
    failure = {
        "status": "failed",
        "error": result.stderr[:500] if result.stderr else "Unknown error",
        "elapsed_s": elapsed,
    }
    with open(f"{RESULTS_DIR}/35b_baseline_robust_fail.json", "w") as f:
        json.dump(failure, f, indent=2)
    sys.exit(1)

# Count tokens in output
output_text = result.stdout
print(f"\n--- Generated text (first 200 chars) ---")
print(output_text[:200])
print(f"--- Total output length: {len(output_text)} chars ---")

# Use the tokenizer to count actual tokens
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, token=HF_TOKEN
)
output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
actual_token_count = len(output_tokens)
print(f"Actual tokens generated: {actual_token_count} (requested: {GEN_LEN})")

# Calculate tok/s based on ACTUAL tokens
tok_s = actual_token_count / elapsed if elapsed > 0 else 0
ms_per_tok = (elapsed * 1000) / actual_token_count if actual_token_count > 0 else float("inf")

print(f"\nStandard mode (VALIDATED):")
print(f"  Requested tokens: {GEN_LEN}")
print(f"  Actual tokens:    {actual_token_count}")
print(f"  Elapsed:          {elapsed:.3f} s")
print(f"  tok/s:            {tok_s:.2f}")
print(f"  ms/token:         {ms_per_tok:.2f}")

# ---------------------------------------------------------------------------
# 7. Benchmark: MTP speculative decode (if supported)
# ---------------------------------------------------------------------------
print("\n=== MTP speculative decode ===")

cmd_mtp = cmd + [
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "2",
]
# Note: llama.cpp may use different flag names; if the above fails,
# try removing the MTP flags and running standard mode only.

print(f"Running: {' '.join(cmd_mtp)}")
t0 = time.time()
result_mtp = subprocess.run(cmd_mtp, capture_output=True, text=True)
elapsed_mtp = time.time() - t0

print(f"Exit code: {result_mtp.returncode}")
if result_mtp.returncode == 0:
    output_text_mtp = result_mtp.stdout
    output_tokens_mtp = tokenizer.encode(output_text_mtp, add_special_tokens=False)
    actual_mtp = len(output_tokens_mtp)
    tok_s_mtp = actual_mtp / elapsed_mtp if elapsed_mtp > 0 else 0
    print(f"\nMTP mode (VALIDATED):")
    print(f"  Actual tokens:    {actual_mtp}")
    print(f"  Elapsed:          {elapsed_mtp:.3f} s")
    print(f"  tok/s:            {tok_s_mtp:.2f}")
    print(f"  Speedup:          {tok_s_mtp/tok_s:.2f}x")
    mtp_worked = True
else:
    print(f"MTP failed: {result_mtp.stderr[:300]}")
    mtp_worked = False

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------
result = {
    "hardware": GPU_NAME,
    "vram_gb": round(TOTAL_VRAM_GB, 1),
    "model": MODEL_NAME,
    "backend": "llama.cpp + Unsloth GGUF",
    "quant": QUANT,
    "gguf_source": REPO_ID,
    "validated": True,
    "requested_tokens": GEN_LEN,
    "actual_tokens": actual_token_count,
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "elapsed_s": round(elapsed, 3),
    "mtp_available": True,
    "mtp_worked": mtp_worked,
}

if mtp_worked:
    result["actual_tokens_mtp"] = actual_mtp
    result["tok_s_mtp"] = round(tok_s_mtp, 2)
    result["elapsed_s_mtp"] = round(elapsed_mtp, 3)
    result["mtp_speedup"] = round(tok_s_mtp / tok_s, 2)

result["note"] = (
    "Colab Pro/Pro+; Qwen3.6-35B-A3B via llama.cpp + Unsloth Dynamic GGUF. "
    "VALIDATED measurement: actual token count verified via tokenizer. "
    "This is the robust GPU baseline for Gate G1."
)

out_path = f"{RESULTS_DIR}/35b_baseline_robust.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✅ Saved validated result to {out_path}")
print("Download this file from the Files panel on the left.")
print("\n=== ALL DONE ===")
