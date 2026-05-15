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
# Backend selection based on VRAM
# ---------------------------------------------------------------------------
# A100 80GB -> transformers FP16 (native)
# A100 40GB -> vLLM (much better memory management, can fit 35B INT4)
# ---------------------------------------------------------------------------
if TOTAL_VRAM_GB >= 80:
    backend = "transformers"
    quant_config = None
    dtype = torch.float16
    strategy = "fp16"
else:
    backend = "vllm"
    strategy = "vllm_fp16"  # vLLM handles quantization internally

print(f"Selected backend: {backend} (VRAM {TOTAL_VRAM_GB:.1f} GB)")

# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
print(f"\n=== Loading {MODEL_NAME} ({backend}) ===")
print("NOTE: First download may take 10-20 minutes for 35B parameters.")

if backend == "transformers":
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
else:
    # vLLM backend
    print("Installing vLLM...")
    rc = os.system("pip install -q vllm")
    if rc != 0:
        raise RuntimeError("vLLM install failed")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        download_dir=None,
    )
    tokenizer = llm.get_tokenizer()

# ---------------------------------------------------------------------------
# 4. Benchmark greedy decode
# ---------------------------------------------------------------------------
print("\n=== Greedy decode benchmark ===")

if backend == "transformers":
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    # Warm-up
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
else:
    # vLLM benchmark
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=GEN_LEN)
    _ = llm.generate(PROMPT, sampling_params)  # warm-up
    t0 = time.time()
    outputs = llm.generate(PROMPT, sampling_params)
    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if hasattr(torch.cuda, "max_memory_allocated") else 0.0

tok_s = GEN_LEN / elapsed
ms_per_tok = (elapsed * 1000) / GEN_LEN

print(f"Backend:          {backend}")
print(f"Strategy:         {strategy}")
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
    "backend": backend,
    "quant_strategy": strategy,
    "tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 2),
    "gen_len": GEN_LEN,
    "elapsed_s": round(elapsed, 3),
    "peak_vram_gb": round(peak_mem_gb, 2),
    "note": f"Colab Pro/Pro+ A100 {TOTAL_VRAM_GB:.0f}GB; Qwen3.6-35B-A3B decode benchmark via {backend}. Gate G1 baseline.",
}

out_path = f"{RESULTS_DIR}/35b_baseline_{backend}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved result to {out_path}")
print("Download this file from the Files panel on the left.")

# ---------------------------------------------------------------------------
# 6. Quick quality sanity
# ---------------------------------------------------------------------------
print("\n=== Quality sanity check ===")
test_prompts = [
    "The capital of France is",
    "Explain quantum computing in one sentence:",
    "Once upon a time",
]

for p in test_prompts:
    if backend == "transformers":
        ids = tokenizer(p, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
    else:
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=20)
        out = llm.generate(p, sp)
        text = out[0].outputs[0].text
    print(f"PROMPT: {p}")
    print(f"OUTPUT: {text}\n")

print("=== ALL DONE ===")
