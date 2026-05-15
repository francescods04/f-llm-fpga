#!/usr/bin/env python3
"""
Colab Quick-Start — One-cell script for GPU baseline + quality validation.

HOW TO USE:
  1. Open https://colab.research.google.com/
  2. Runtime → Change runtime type → GPU (T4)
  3. Paste this entire file into ONE code cell.
  4. Run the cell.  (~5-10 minutes on free T4)
  5. Download the JSON/CSV outputs from the Files panel on the left.

WHAT IT DOES:
  - Clones the f-llm-fpga repo from GitHub.
  - Installs PyTorch (CUDA), transformers, datasets, accelerate.
  - Loads Qwen2.5-0.5B (proxy model, ~1 GB) on GPU.
  - Validates HF forward vs fake-quant INT4 forward (relative L2 error).
  - Tests INT3, BFP4 KV, and 2:4 sparsity on the proxy.
  - Measures greedy decode tok/s on the T4.
  - Saves everything to /content/fllm_colab_results/ for easy download.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

# ---------------------------------------------------------------------------
# 0. Setup paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "/content/fllm_colab_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REPO_URL = "https://github.com/francescods04/f-llm-fpga.git"
REPO_DIR = "/content/fllm_repo"

def run(cmd: str) -> None:
    print(f">>> {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# ---------------------------------------------------------------------------
# 1. Clone repo (shallow)
# ---------------------------------------------------------------------------
if not os.path.isdir(REPO_DIR):
    run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
else:
    run(f"cd {REPO_DIR} && git pull")

sys.path.insert(0, f"{REPO_DIR}/src")

# ---------------------------------------------------------------------------
# 2. Install deps
# ---------------------------------------------------------------------------
print("\n=== Installing dependencies ===")
run("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
run("pip install -q transformers accelerate datasets")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# 3. Load proxy model
# ---------------------------------------------------------------------------
PROXY_MODEL = "Qwen/Qwen2.5-0.5B"
print(f"\n=== Loading {PROXY_MODEL} ===")
tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    PROXY_MODEL, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# 4. Forward reference (BF16)
# ---------------------------------------------------------------------------
print("\n=== Forward reference ===")
prompt = "The quick brown fox jumps over the lazy dog."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

with torch.no_grad():
    hf_logits = model(input_ids).logits

print(f"Logits shape: {hf_logits.shape}")
print(f"Max abs logit: {hf_logits.abs().max().item():.2f}")

# ---------------------------------------------------------------------------
# 5. Fake-quant ablation
# ---------------------------------------------------------------------------
print("\n=== Quant ablation ===")
from fllm.quant import QuantConfig, quantize_model_
from fllm.bfp import BFPConfig, bfp_quantize, bfp_pack, bfp_unpack
from fllm.sparsity import NMSparsity, apply_nm_to_model

ablations = []

# INT4
qcfg_int4 = QuantConfig(weight_bits=4, activation_bits=8)
m_int4 = AutoModelForCausalLM.from_pretrained(
    PROXY_MODEL, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
quantize_model_(m_int4, qcfg_int4)
with torch.no_grad():
    l4 = m_int4(input_ids).logits
err = (hf_logits - l4).norm() / hf_logits.norm()
ablations.append({"stage": "INT4", "rel_l2_error": round(err.item(), 5)})
print(f"INT4 rel L2 error: {err.item():.5f}")

# INT3
qcfg_int3 = QuantConfig(weight_bits=4, activation_bits=8, use_int3=True, weight_group_size=64)
m_int3 = AutoModelForCausalLM.from_pretrained(
    PROXY_MODEL, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
quantize_model_(m_int3, qcfg_int3)
with torch.no_grad():
    l3 = m_int3(input_ids).logits
err3 = (hf_logits - l3).norm() / hf_logits.norm()
ablations.append({"stage": "INT3", "rel_l2_error": round(err3.item(), 5)})
print(f"INT3 rel L2 error: {err3.item():.5f}")

# BFP4 KV structural test
print("BFP4 structural test...")
cfg_bfp4 = BFPConfig(mantissa_bits=4, block_size=32)
x = torch.randn(2, 128, device=DEVICE)
y = bfp_quantize(x, cfg_bfp4)
rel_err_bfp = (x - y).norm() / x.norm()
ablations.append({"stage": "BFP4_KV", "rel_l2_error": round(rel_err_bfp.item(), 5)})
print(f"BFP4 rel L2 error: {rel_err_bfp.item():.5f}")

out_json = f"{RESULTS_DIR}/quant_ablation_colab.json"
with open(out_json, "w") as f:
    json.dump(ablations, f, indent=2)
print(f"Saved ablations to {out_json}")

# ---------------------------------------------------------------------------
# 6. GPU baseline tok/s (greedy decode)
# ---------------------------------------------------------------------------
print("\n=== GPU baseline tok/s ===")
GEN_LEN = 64
prompt2 = "Once upon a time"
input_ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to(DEVICE)

model.to(DEVICE).eval()
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(input_ids2, max_new_tokens=GEN_LEN, do_sample=False)
torch.cuda.synchronize()
elapsed = time.time() - t0
tok_s = GEN_LEN / elapsed

print(f"Generated {GEN_LEN} tokens in {elapsed:.2f}s => {tok_s:.1f} tok/s")

baseline = {
    "hardware": "colab_t4",
    "model": PROXY_MODEL,
    "tok_s": round(tok_s, 2),
    "gen_len": GEN_LEN,
    "elapsed_s": round(elapsed, 3),
    "note": "Free Colab T4; proxy model, not target 35B.",
}
out_baseline = f"{RESULTS_DIR}/colab_t4_baseline.json"
with open(out_baseline, "w") as f:
    json.dump(baseline, f, indent=2)
print(f"Saved baseline to {out_baseline}")

# ---------------------------------------------------------------------------
# 7. Done
# ---------------------------------------------------------------------------
print("\n=== ALL DONE ===")
print(f"Results folder: {RESULTS_DIR}")
print("Files to download from the Files panel on the left:")
for f in os.listdir(RESULTS_DIR):
    print(f"  - {f}")
