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
4. **Full-model canary:** one short prompt produces coherent output with a
   bounded context and token count.

No pull request or claim of V4 support until gate 4 passes.

## Deferred

- DSpark speculative decoding.
- Long-context validation.
- Expert caching, prefetch scheduling, and production serving.
- CPU expert execution.
