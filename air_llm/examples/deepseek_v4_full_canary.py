"""Run a bounded full-depth DeepSeek V4 generation canary on CUDA.

The prompt uses the model release's native encoder. It runs twice in one process so the output
reports both cold Triton compilation time and warm generation time.
"""

import gc
import json
import resource
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from types import MethodType

import torch

from airllm.airllm_deepseek_v4 import AirLLMDeepseekV4


MIN_FREE_GIB = 80
ALLOCATOR_FRACTION = 0.5
PROMPT = "Reply with exactly: Paris"
MAX_NEW_TOKENS = 4


def main():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free_before, total = torch.cuda.mem_get_info(device)
    if free_before < MIN_FREE_GIB * 2**30:
        raise SystemExit(
            f"ABORT: requires {MIN_FREE_GIB} GiB free VRAM; "
            f"only {free_before / 2**30:.2f} GiB is available"
        )
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION, device.index)

    sys.path.insert(0, str(args.model_path / "encoding"))
    from encoding_dsv4 import encode_messages

    model = AirLLMDeepseekV4(
        args.model_path,
        device=str(device),
        dtype=torch.bfloat16,
        max_seq_len=64,
        prefetching=False,
    )
    prompt = encode_messages([{"role": "user", "content": PROMPT}], thinking_mode="chat")
    prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(prompt_ids)

    state = {"run": 0, "passes": 0, "expert_loads": 0}
    pass_records = []
    original_experts = model._run_streamed_experts

    def counted_experts(self, module, hidden_states, top_k_index, top_k_weights):
        state["expert_loads"] += int(torch.unique(top_k_index).numel())
        return original_experts(module, hidden_states, top_k_index, top_k_weights)

    def completed_sweep(module, hook_args, output):
        state["passes"] += 1
        pass_records.append(
            {
                "run": state["run"],
                "pass": state["passes"],
                "distinct_expert_loads": state["expert_loads"],
            }
        )
        state["expert_loads"] = 0

    model._run_streamed_experts = MethodType(counted_experts, model)
    model.model.model.layers[-1].register_forward_hook(completed_sweep)

    runs = []
    reference = None
    for run in (1, 2):
        state.update(run=run, passes=0, expert_loads=0)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize(device)
        completion_ids = generated[0, prompt_ids.shape[-1] :]
        completion = model.tokenizer.decode(completion_ids, skip_special_tokens=False)
        identical = reference is None or bool(torch.equal(reference, generated))
        if reference is None:
            reference = generated.detach().clone()
        runs.append(
            {
                "run": run,
                "seconds": round(time.perf_counter() - started, 3),
                "passes": state["passes"],
                "generated_token_ids": completion_ids.tolist(),
                "completion": completion,
                "identical_to_first": identical,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3),
            }
        )
        del generated, completion_ids
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "status": "pass",
        "prompt": PROMPT,
        "encoded_prompt_tokens": int(prompt_ids.shape[-1]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "free_before_gib": round(free_before / 2**30, 3),
        "allocator_cap_gib": round(total * ALLOCATOR_FRACTION / 2**30, 3),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
        "runs": runs,
        "pass_records": pass_records,
    }
    print(json.dumps(result, indent=2))

    if not all(run["identical_to_first"] for run in runs):
        raise SystemExit("FAIL: greedy generation changed across runs")
    if not runs[-1]["completion"].startswith("Paris"):
        raise SystemExit(f"FAIL: unexpected completion {runs[-1]['completion']!r}")


if __name__ == "__main__":
    main()
