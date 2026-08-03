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

from airllm import airllm_deepseek_v4 as deepseek_v4_module
from airllm.airllm_deepseek_v4 import AirLLMDeepseekV4


MIN_FREE_GIB = 80
ALLOCATOR_FRACTION = 0.5
PROMPT = "Reply with exactly: Paris"
MAX_NEW_TOKENS = 4


def main():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--expect-prefix", default="Paris")
    parser.add_argument("--prefetching", action="store_true")
    parser.add_argument("--expert-cache-size", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
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
        prefetching=args.prefetching,
        expert_cache_size=args.expert_cache_size,
    )
    prompt = encode_messages([{"role": "user", "content": args.prompt}], thinking_mode="chat")
    prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(prompt_ids)

    state = {
        "run": 0,
        "passes": 0,
        "expert_loads": 0,
        "expert_routes": set(),
        "previous_expert_routes": None,
        "sweep_started": None,
        "expert_read_seconds": 0.0,
        "non_expert_load_seconds": 0.0,
        "expert_host_seconds": 0.0,
        "expert_cuda_events": [],
    }
    pass_records = []
    original_experts = model._run_streamed_experts

    def counted_experts(self, module, hidden_states, top_k_index, top_k_weights):
        routed = {int(expert_idx) for expert_idx in torch.unique(top_k_index).tolist()}
        state["expert_loads"] += len(routed)
        state["expert_routes"].update(
            (module._airllm_layer_idx, expert_idx) for expert_idx in routed)
        started = time.perf_counter()
        result = original_experts(module, hidden_states, top_k_index, top_k_weights)
        state["expert_host_seconds"] += time.perf_counter() - started
        return result

    if args.profile:
        original_subset = deepseek_v4_module.load_layer_subset

        def timed_subset(local_path, layer_name, keys):
            started = time.perf_counter()
            result = original_subset(local_path, layer_name, keys)
            if any('.experts.' in key for key in keys):
                state["expert_read_seconds"] += time.perf_counter() - started
            return result

        deepseek_v4_module.load_layer_subset = timed_subset

        original_streamed_layer = model._load_streamed_layer

        def timed_streamed_layer(self, idx):
            started = time.perf_counter()
            result = original_streamed_layer(idx)
            state["non_expert_load_seconds"] += time.perf_counter() - started
            return result

        model._load_streamed_layer = MethodType(timed_streamed_layer, model)

        original_linear = model._expert_linear

        def timed_linear(self, inputs, tensors, projection):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            result = original_linear(inputs, tensors, projection)
            finished.record()
            state["expert_cuda_events"].append((started, finished))
            return result

        model._expert_linear = MethodType(timed_linear, model)

    def completed_sweep(module, hook_args, output):
        if args.profile:
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        previous = state["previous_expert_routes"]
        reused = len(state["expert_routes"] & previous) if previous is not None else None
        sweep_seconds = now - state["sweep_started"]
        state["passes"] += 1
        record = {
            "run": state["run"],
            "pass": state["passes"],
            "distinct_expert_loads": state["expert_loads"],
            "expert_routes_reused": reused,
            "expert_route_reuse_fraction": (
                round(reused / len(state["expert_routes"]), 3) if reused is not None else None
            ),
            "seconds_since_previous_sweep": round(sweep_seconds, 3),
        }
        if args.profile:
            expert_cuda_seconds = sum(
                started.elapsed_time(finished)
                for started, finished in state["expert_cuda_events"]
            ) / 1000
            record["profile"] = {
                "expert_read_seconds": round(state["expert_read_seconds"], 3),
                "non_expert_load_seconds": round(state["non_expert_load_seconds"], 3),
                "expert_host_seconds": round(state["expert_host_seconds"], 3),
                "expert_cuda_seconds": round(expert_cuda_seconds, 3),
            }
        pass_records.append(record)
        state["expert_loads"] = 0
        state["previous_expert_routes"] = state["expert_routes"]
        state["expert_routes"] = set()
        state["sweep_started"] = now
        state["expert_read_seconds"] = 0.0
        state["non_expert_load_seconds"] = 0.0
        state["expert_host_seconds"] = 0.0
        state["expert_cuda_events"] = []

    model._run_streamed_experts = MethodType(counted_experts, model)
    model.model.model.layers[-1].register_forward_hook(completed_sweep)

    runs = []
    reference = None
    for run in range(1, args.runs + 1):
        state.update(
            run=run,
            passes=0,
            expert_loads=0,
            expert_routes=set(),
            previous_expert_routes=None,
            sweep_started=time.perf_counter(),
            expert_read_seconds=0.0,
            non_expert_load_seconds=0.0,
            expert_host_seconds=0.0,
            expert_cuda_events=[],
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
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
        "prompt": args.prompt,
        "encoded_prompt_tokens": int(prompt_ids.shape[-1]),
        "max_new_tokens": args.max_new_tokens,
        "prefetching": args.prefetching,
        "expert_cache_size": args.expert_cache_size,
        "profile": args.profile,
        "free_before_gib": round(free_before / 2**30, 3),
        "allocator_cap_gib": round(total * ALLOCATOR_FRACTION / 2**30, 3),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
        "runs": runs,
        "pass_records": pass_records,
    }
    print(json.dumps(result, indent=2))

    if not all(run["identical_to_first"] for run in runs):
        raise SystemExit("FAIL: greedy generation changed across runs")
    if args.expect_prefix and not runs[-1]["completion"].startswith(args.expect_prefix):
        raise SystemExit(f"FAIL: unexpected completion {runs[-1]['completion']!r}")


if __name__ == "__main__":
    main()
