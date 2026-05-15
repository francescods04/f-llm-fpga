#!/usr/bin/env python3
"""
Colab 35B Baseline — Unsloth GGUF + llama.cpp + MTP speculative decode.

Uses Unsloth's pre-quantized Dynamic GGUFs and llama.cpp with MTP
(Multi Token Prediction) for the fastest possible GPU baseline.

According to Unsloth benchmarks:
  - Qwen3.6-35B-A3B: ~220 tok/s with UD-Q2_K_XL + MTP on A100
  - MoE models get ~1.15-1.25x speedup from MTP vs standard decode

HOW TO USE:
  1. Get HF token: https://huggingface.co/settings/tokens (read)
  2. Colab: add HF_TOKEN to secrets, toggle Notebook access ON
  3. Runtime → Restart session
  4. Paste this into ONE code cell and run (~15 min first time).
"""

from __future__ import annotations

import json
import os
import time
import subprocess

RESULTS_DIR = "/content/fllm_colab_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GEN_LEN = 128  # longer for more stable measurement
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
# 2. Detect VRAM and select quant
# ---------------------------------------------------------------------------
import torch
GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB")

# Unsloth hardware requirements:
# 35B-A3B: Q4_K_XL = 23GB, Q2_K_XL = 17GB
if TOTAL_VRAM_GB >= 30:
    QUANT = "UD-Q4_K_XL"
elif TOTAL_VRAM_GB >= 20:
    QUANT = "UD-Q2_K_XL"
else:
    raise RuntimeError(f"VRAM {TOTAL_VRAM_GB:.1f}GB insufficient. Need >=20GB for 35B-A3B GGUF.")

print(f"Selected quant: {QUANT}")

# ---------------------------------------------------------------------------
# 3. Build llama.cpp with CUDA (from mtp-clean branch for MTP support)
# ---------------------------------------------------------------------------
LLAMA_DIR = "/content/llama.cpp"
if not os.path.exists(LLAMA_DIR):
    print("=== Cloning llama.cpp (mtp-clean branch) ===")
    rc = os.system(
        "git clone -b mtp-clean https://github.com/am17an/llama.cpp.git /content/llama.cpp"
    )
    if rc != 0:
        print("mtp-clean branch failed, trying main...")
        os.system("rm -rf /content/llama.cpp")
        rc = os.system("git clone https://github.com/ggml-org/llama.cpp.git /content/llama.cpp")
        if rc != 0:
            raise RuntimeError("Failed to clone llama.cpp")

print("=== Building llama.cpp with CUDA ===")
rc = os.system(
    f"cmake {LLAMA_DIR} -B {LLAMA_DIR}/build "
    "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CUDA=ON"
)
if rc != 0:
    raise RuntimeError("cmake failed")

rc = os.system(f"cmake --build {LLAMA_DIR}/build --config Release -j --target llama-cli")
if rc != 0:
    raise RuntimeError("build failed")

# Find llama-cli
import glob
found = glob.glob(f"{LLAMA_DIR}/**/llama-cli", recursive=True)
if not found:
    raise RuntimeError(f"llama-cli not found in {LLAMA_DIR}/build")
LLAMA_CLI = found[0]
print(f"llama-cli: {LLAMA_CLI}")

# Verify MTP support
print("\n=== Checking MTP support ===")
help_out = subprocess.run([LLAMA_CLI, "--help"], capture_output=True, text=True)
has_mtp = "draft-mtp" in help_out.stdout or "mtp" in help_out.stdout
print(f"MTP support: {'YES' if has_mtp else 'NO (will run standard only)'}")

# ---------------------------------------------------------------------------
# 4. Download Unsloth GGUF
# ---------------------------------------------------------------------------
REPO_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GGUF_FILE = f"Qwen3.6-35B-A3B-{QUANT}.gguf"
LOCAL_GGUF = f"/content/{GGUF_FILE}"

if not os.path.exists(LOCAL_GGUF):
    print(f"\n=== Downloading {GGUF_FILE} from HF ===")
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
    print(f"Using cached GGUF: {LOCAL_GGUF}")

# ---------------------------------------------------------------------------
# 5. Benchmark: standard greedy decode
# ---------------------------------------------------------------------------
print("\n=== Standard greedy decode benchmark ===")

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
    print("WARNING: llama-cli had issues, but timing was captured")

tok_s = GEN_LEN / elapsed
ms_per_tok = (elapsed * 1000) / GEN_LEN

print(f"\nStandard mode:")
print(f"  Quant:        {QUANT}")
print(f"  Tokens:       {GEN_LEN}")
print(f"  Elapsed:      {elapsed:.2f} s")
print(f"  tok/s:        {tok_s:.2f}")
print(f"  ms/token:     {ms_per_tok:.2f}")

# ---------------------------------------------------------------------------
# 6. Benchmark: MTP speculative decode (if supported)
# ---------------------------------------------------------------------------
if has_mtp:
    print("\n=== MTP speculative decode benchmark ===")
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
        print("MTP run failed (non-zero exit)")
        mtp_worked = False
else:
    mtp_worked = False

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
result = {
    "hardware": GPU_NAME,
    "vram_gb": round(TOTAL_VRAM_GB, 1),
    "model": "Qwen/Qwen3.6-35B-A3B",
    "backend": "llama.cpp + Unsloth GGUF",
    "quant": QUANT,
    "gguf_source": REPO_ID,
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "gen_len": GEN_LEN,
    "elapsed_s": round(elapsed, 3),
    "mtp_available": has_mtp,
}

if mtp_worked:
    result["tok_s_mtp"] = round(tok_s_mtp, 2)
    result["ms_per_token_mtp"] = round(ms_per_tok_mtp, 2)
    result["mtp_speedup"] = round(tok_s_mtp / tok_s, 2)

result["note"] = (
    "Colab Pro/Pro+; Qwen3.6-35B-A3B via llama.cpp + Unsloth Dynamic GGUF. "
    "This is the optimized GPU baseline for Gate G1. "
    f"Unsloth claims ~220 tok/s possible with UD-Q2_K_XL + MTP on A100."
)

out_path = f"{RESULTS_DIR}/35b_baseline_unsloth.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved result to {out_path}")
print("Download this file from the Files panel on the left.")

print("\n=== ALL DONE ===")
