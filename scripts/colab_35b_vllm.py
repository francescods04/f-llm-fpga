#!/usr/bin/env python3
"""
Colab 35B Baseline (vLLM fallback) — Run Qwen3.6-35B-A3B via vLLM.

Use this if transformers from git still does not recognize 'qwen3_5_moe'.
vLLM typically adds new Qwen architectures much faster than transformers.

HOW TO USE:
  1. Get HF token: https://huggingface.co/settings/tokens (read)
  2. Colab: add HF_TOKEN to secrets (🔑), toggle Notebook access ON
  3. Runtime → Restart session
  4. Paste this into ONE code cell and run.
  
IMPORTANT: If you see 'XPU_KERNEL_FORMAT' error, do Runtime → Restart session.
"""

from __future__ import annotations

import json
import os
import time
import sys

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
# Install vLLM (let it install its own compatible torch)
# ---------------------------------------------------------------------------
INSTALL_MARKER = "/content/.vllm_installed"

if not os.path.exists(INSTALL_MARKER):
    print("=== Installing vLLM ===")
    # Remove existing torch to avoid version conflicts (vLLM needs CUDA 12 torch)
    print("Removing potentially conflicting torch installation...")
    os.system("pip uninstall -y torch torchvision torchaudio 2>/dev/null")
    # Install vLLM — this will pull the correct torch version
    rc = os.system("pip install -q vllm")
    if rc != 0:
        raise RuntimeError("vLLM install failed")

    # Save marker and ask for restart
    with open(INSTALL_MARKER, "w") as f:
        f.write("done")
    
    print("\n" + "="*60)
    print("vLLM INSTALLED SUCCESSFULLY")
    print("="*60)
    print("\nIMPORTANT: You MUST restart the Python kernel now.")
    print("Colab: Runtime → Restart session")
    print("Then re-run this same cell.")
    print("="*60 + "\n")
    
    # Exit so user must restart
    sys.exit(0)
else:
    print("vLLM already installed (marker found). Proceeding with benchmark...")

# ---------------------------------------------------------------------------
# Main execution (runs after restart)
# ---------------------------------------------------------------------------
from vllm import LLM, SamplingParams
import torch

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB")

# ---------------------------------------------------------------------------
# Load model with vLLM
# ---------------------------------------------------------------------------
print(f"\n=== Loading {MODEL_NAME} via vLLM ===")
print("NOTE: First download may take 10-20 minutes for 35B parameters.")

# For A100 80GB we can fit FP16; for 40GB use quantization="fp8" or "awq"
quantization = None if TOTAL_VRAM_GB >= 80 else "fp8"

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    quantization=quantization,
    trust_remote_code=True,
    download_dir=None,  # uses default HF cache
)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
print("\n=== Greedy decode benchmark ===")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=GEN_LEN)

# Warm-up
_ = llm.generate(PROMPT, sampling_params)

# Measure
t0 = time.time()
outputs = llm.generate(PROMPT, sampling_params)
elapsed = time.time() - t0

tok_s = GEN_LEN / elapsed
ms_per_tok = (elapsed * 1000) / GEN_LEN

print(f"Quantization:    {quantization or 'fp16'}")
print(f"Tokens generated: {GEN_LEN}")
print(f"Elapsed time:     {elapsed:.2f} s")
print(f"Throughput:       {tok_s:.2f} tok/s")
print(f"ms/token:         {ms_per_tok:.2f}")
print(f"Generated text:   {outputs[0].outputs[0].text[:200]}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
result = {
    "hardware": GPU_NAME,
    "vram_gb": round(TOTAL_VRAM_GB, 1),
    "model": MODEL_NAME,
    "backend": "vllm",
    "quantization": quantization or "fp16",
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "gen_len": GEN_LEN,
    "elapsed_s": round(elapsed, 3),
    "note": "Colab Pro/Pro+ A100; Qwen3.6-35B-A3B via vLLM. Gate G1 baseline.",
}

out_path = f"{RESULTS_DIR}/35b_baseline_vllm.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved result to {out_path}")
print("Download this file from the Files panel on the left.")
