#!/usr/bin/env python3
"""
Colab 35B Baseline — Run Qwen3.6-35B-A3B on Colab Pro/Pro+ GPU.

IMPORTANT: This model is private/gated on HuggingFace.
You MUST add your HuggingFace token to Colab secrets before running.

HOW TO USE:
  1. Get a HuggingFace token: https://huggingface.co/settings/tokens (scope: read)
  2. In Colab, click the 🔑 Secrets icon on the left panel.
  3. Add a secret: Name = HF_TOKEN, Value = your token.
  4. Toggle "Notebook access" ON.
  5. Runtime → Restart session.
  6. Open https://colab.research.google.com
  7. Runtime → Change runtime type → GPU (A100 if available)
  8. Paste this script into one code cell and run.

WHAT IT DOES:
  - Detects GPU VRAM and picks quantization strategy (none / 8-bit / 4-bit).
  - Downloads the model from HuggingFace (first run ~10-20 min for 35B).
  - Runs greedy decode benchmark (batch=1, 64 new tokens).
  - Logs tok/s, ms/tok, GPU memory peak.
  - Saves JSON to /content/fllm_colab_results/ for download.

If you do NOT have HF access, switch to a public proxy model by setting
MODEL_NAME to "Qwen/Qwen2.5-14B" or "Qwen/Qwen2-57B-A14B".
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

RESULTS_DIR = "/content/fllm_colab_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIGURATION — change these
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"   # <-- real HF id (private/gated, needs token)
GEN_LEN = 64
PROMPT = "The future of artificial intelligence is"

# ---------------------------------------------------------------------------
# HuggingFace token (read from Colab secrets or env var)
# ---------------------------------------------------------------------------
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass

if HF_TOKEN is None:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    print("HF_TOKEN loaded from secrets/environment.")
else:
    print("WARNING: HF_TOKEN not found. If the model is private/gated, download will fail.")
    print("Add HF_TOKEN to Colab secrets (left panel 🔑) and restart the session.")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def run(cmd: str) -> None:
    print(f">>> {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
print("=== Installing dependencies ===")
run("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
# transformers stable may not yet support qwen3_5_moe; install from source
run("pip install -q git+https://github.com/huggingface/transformers.git")
run("pip install -q accelerate bitsandbytes")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Change runtime to GPU (A100 preferred).")

DEVICE = "cuda"
GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM_GB:.1f} GB")

# ---------------------------------------------------------------------------
# 2. Quantization strategy based on VRAM
# ---------------------------------------------------------------------------
# FP16 model = ~70 GB. INT4 = ~17.5 GB. INT8 = ~35 GB.
# We need VRAM >= model_size + activations + KV cache overhead (~5 GB).
if TOTAL_VRAM_GB >= 85:
    quant_config = None
    dtype = torch.float16
    strategy = "fp16"
elif TOTAL_VRAM_GB >= 45:
    # 8-bit via bitsandbytes
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    dtype = torch.float16
    strategy = "int8"
else:
    # 4-bit via bitsandbytes
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    dtype = torch.float16
    strategy = "int4"

print(f"Selected strategy: {strategy} (VRAM {TOTAL_VRAM_GB:.1f} GB)")

# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
print(f"\n=== Loading {MODEL_NAME} ({strategy}) ===")
print("NOTE: First download may take 10-20 minutes for 35B parameters.")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    model.eval()
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("If the model is not yet public, update MODEL_NAME to the correct HF id.")
    raise

# ---------------------------------------------------------------------------
# 4. Benchmark greedy decode
# ---------------------------------------------------------------------------
print("\n=== Greedy decode benchmark ===")
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

# Warm-up (graph compilation / cache allocation)
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=2, do_sample=False)
torch.cuda.synchronize()

# Measure
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=GEN_LEN, do_sample=False)
torch.cuda.synchronize()
elapsed = time.time() - t0
peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

tok_s = GEN_LEN / elapsed
ms_per_tok = (elapsed * 1000) / GEN_LEN

print(f"Strategy:        {strategy}")
print(f"Tokens generated: {GEN_LEN}")
print(f"Elapsed time:     {elapsed:.2f} s")
print(f"Throughput:       {tok_s:.2f} tok/s")
print(f"ms/token:         {ms_per_tok:.2f}")
print(f"Peak VRAM:        {peak_mem_gb:.2f} GB")

# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
result = {
    "hardware": GPU_NAME,
    "vram_gb": round(TOTAL_VRAM_GB, 1),
    "model": MODEL_NAME,
    "quant_strategy": strategy,
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "gen_len": GEN_LEN,
    "elapsed_s": round(elapsed, 3),
    "peak_vram_gb": round(peak_mem_gb, 2),
    "note": "Colab Pro/Pro+; Qwen3.6-35B-A3B decode benchmark. "
            "This is the real target model baseline for Gate G1.",
}

out_path = f"{RESULTS_DIR}/35b_baseline_{strategy}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved result to {out_path}")
print("Download this file from the Files panel on the left.")

# ---------------------------------------------------------------------------
# 6. Quick quality sanity (single prompt consistency check)
# ---------------------------------------------------------------------------
print("\n=== Quality sanity check ===")
test_prompts = [
    "The capital of France is",
    "Explain quantum computing in one sentence:",
    "Once upon a time",
]

for p in test_prompts:
    ids = tokenizer(p, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"PROMPT: {p}")
    print(f"OUTPUT: {text}\n")

print("=== ALL DONE ===")
