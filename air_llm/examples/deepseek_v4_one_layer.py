"""Run a bounded, one-decoder-layer DeepSeek V4 correctness canary on CUDA.

Install the optional kernel runtime first:

    pip install -e './air_llm[deepseek-v4]'

The full architecture is created on the meta device so its streaming hooks match a normal AirLLM
run, but only decoder layer 0 is left in the execution loop. This exercises embeddings, attention,
routing, the shared expert, six streamed routed experts, the final norm, and the language head.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from airllm.airllm_deepseek_v4 import AirLLMDeepseekV4


MIN_FREE_GIB = 20
ALLOCATOR_FRACTION = 0.18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free_before, total = torch.cuda.mem_get_info(device)
    minimum_free = MIN_FREE_GIB * 2**30
    if free_before < minimum_free:
        raise SystemExit(
            f"ABORT: requires {MIN_FREE_GIB} GiB free VRAM; "
            f"only {free_before / 2**30:.2f} GiB is available"
        )

    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION, device.index)
    torch.cuda.empty_cache()

    init_started = time.perf_counter()
    model = AirLLMDeepseekV4(
        args.model_path,
        device=str(device),
        dtype=torch.bfloat16,
        max_seq_len=8,
        prefetching=False,
    )
    init_seconds = time.perf_counter() - init_started

    # Retain the real layer and its installed AirLLM hooks; shorten only the model's execution loop.
    decoder = model.model.model
    decoder.layers = torch.nn.ModuleList([decoder.layers[0]])
    model.config.num_hidden_layers = 1
    decoder.config.num_hidden_layers = 1

    input_ids = torch.tensor([[1]], device=device, dtype=torch.long)
    torch.cuda.reset_peak_memory_stats(device)
    forward_started = time.perf_counter()
    with torch.no_grad():
        output = model.model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started

    experts = decoder.layers[0].mlp.experts._airllm_last_experts
    logits = output.logits
    result = {
        "status": "pass",
        "scope": "embed + decoder layer 0 + norm + lm_head",
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "free_before_gib": round(free_before / 2**30, 3),
        "allocator_cap_gib": round(total * ALLOCATOR_FRACTION / 2**30, 3),
        "init_seconds": round(init_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3),
        "routed_experts": [int(expert) for expert in experts],
        "routed_expert_count": len(experts),
        "logits": {
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "finite": bool(torch.isfinite(logits).all().item()),
            "abs_mean": float(logits.abs().mean().item()),
        },
    }
    print(json.dumps(result, indent=2))

    if len(experts) != model.config.num_experts_per_tok:
        raise SystemExit(
            f"FAIL: expected {model.config.num_experts_per_tok} routed experts, got {len(experts)}"
        )
    if not result["logits"]["finite"]:
        raise SystemExit("FAIL: non-finite logits")

    del output, logits, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
