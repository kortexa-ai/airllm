# DeepSeek V4 Flash support plan

Goal: make AirLLM load and execute DeepSeek-V4-Flash-0731 by keeping the
ordinary model weights resident or layer-streamed while loading only the
routed experts selected by each V4 MoE layer.

## Scope

- Add a dedicated `DeepseekV4ForCausalLM` adapter.
- Read the released checkpoint's native `layers.*`, `embed.*`, `norm.*`, and
  `head.*` tensor layout without duplicating the 167 GB checkpoint.
- Override the stacked Transformers expert implementation so selected experts
  can be loaded individually from safetensors.
- Prove routing, selective loading, and execution with a tiny synthetic V4
  checkpoint before using the real model.
- Keep DSpark/speculative decoding and performance tuning out of the first
  correctness milestone.

## Gates

1. **Local structural test — passed:** all 72,317 released checkpoint keys were
   checked against a quantizer-prepared Transformers V4 model. Every non-expert,
   non-MTP key maps to a real parameter, and all 43 layer shards are directly
   linkable rather than copied.
2. **Local execution test — passed:** a tiny native-format V4 model produces
   exactly the same logits as Transformers while loading only the experts selected
   by its router.
3. **One-layer smarty canary — passed:** native packed FP4 expert tensors were read,
   transferred, and executed on SM120. A one-token layer-0 forward routed to exactly six
   experts and produced finite BF16 logits with 1,054 MiB peak allocation / 1,066 MiB peak
   reservation. The first forward took 65.5 seconds including Triton kernel fetch/compile;
   DeepGEMM currently declines SM120. Reproduce with
   `air_llm/examples/deepseek_v4_one_layer.py` after installing the `deepseek-v4` extra.
4. **Full-model canary — passed:** the official chat encoder produced a nine-token
   `Reply with exactly: Paris` prompt, and a 43-layer greedy run returned `Paris` plus EOS.
   Cold generation took 47.48 seconds and the identical warm run took 13.27 seconds.
   Prefill loaded 1,305 distinct routed experts across the layers; one-token decode loaded
   exactly 258 (six per layer). Peak allocation/reservation was 1,045/1,050 MiB and peak
   process RSS was 3.12 GiB. A separate sustained-decode run correctly returned
   `1, 2, 3, 4, 5` plus EOS over 14 sweeps; its 12 measured warm decode sweeps averaged
   5.76 seconds/token, held at 258 expert loads each, and kept free VRAM flat. Reproduce
   the basic full-model gate with `air_llm/examples/deepseek_v4_full_canary.py`.

Gate 4 is complete. Keep the upstream pull request deferred until the optimized path has broader
prompt/context coverage and user-facing documentation.

## Performance pass

The fresh control measured 5.21 seconds/token. The completed performance pass produced:

- Skipping V4's unnecessary per-layer GC and CUDA cache purge: 1.92 seconds/token.
- Reading all routed experts through one safetensors open per layer: 1.73 seconds/token.
- A single-token routing fast path: 1.69 seconds/token at the minimum ~1 GiB allocation.
- Keeping 8.5 GiB of ordinary/non-expert weights resident: 0.63 seconds/token.
- Resident ordinary weights plus a 64-expert-per-layer LRU: 0.272 seconds/token (3.68 tokens/s)
  averaged across 13 sustained decode sweeps. The last five cache-warm tokens averaged 0.225
  seconds/token (4.44 tokens/s). Peak allocation/reservation was 36.0/36.0 GiB and host RSS was
  3.02 GiB; output remained `1, 2, 3, 4, 5` plus EOS.

Existing next-layer prefetching regressed to 1.85 seconds/token. A six-expert batched Triton path
regressed to 1.81 seconds/token, and fusing each expert's gate/up projections regressed the final
configuration to 0.290 seconds/token, so all three experiments were left out. Cache size 72 was
indistinguishable from 64. Further material gains now require kernel/graph work or broader serving
changes rather than more file-I/O tuning.

## Deferred

- DSpark speculative decoding.
- Long-context validation.
- Production serving.
- CPU expert execution.
