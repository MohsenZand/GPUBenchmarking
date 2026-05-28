"""
LLM Benchmark — throughput, TTFT, and quantization.

Quantization notes:
  - bf16/fp16: no weight quantization; dtype controls compute precision.
  - fp8: on-the-fly W8A8 via vLLM quantization="fp8" (H100/B200/B300).
  - fp4 (NV-FP4): requires pre-quantized checkpoints incompatible with the
    standard Llama-3 HF weights and the current vLLM/modelopt version pairing;
    not included in this benchmark.

Other notes:
  - Two warmup passes before any timing (avoids CUDA graph compile in measurement)
  - TTFT measured via max_tokens=1 on a single request (real prefill latency)
  - Peak TPS recorded across a concurrent-request sweep (not a single fixed batch)
  - VRAM_Util measured from pynvml (was hardcoded to the engine request knob)
  - 150 prompts covering short/medium/long lengths for realistic load
"""
import time, csv, os, argparse, gc
import torch
import pynvml
from vllm import LLM, SamplingParams


# ── Prompt pool: 150 requests, three length tiers ──────────────────────────
_SHORT  = "What is the role of GPUs in modern AI workloads?"

_MEDIUM = (
    "Explain in technical detail how transformer self-attention works, "
    "including the mathematical formulation of queries, keys, and values, "
    "and how multi-head attention extends this mechanism."
)

_LONG = (
    "Describe the complete history of deep learning from the 1950s through "
    "today, covering perceptrons, backpropagation, convolutional neural networks, "
    "recurrent networks, LSTMs, attention mechanisms, and large language models. "
    "Include key papers, researchers, and the hardware advances that enabled each era. "
) * 3

PROMPT_POOL = [_SHORT] * 60 + [_MEDIUM] * 60 + [_LONG] * 30  # 150 total


# ── VRAM helpers ────────────────────────────────────────────────────────────
def _used_vram_gb(handles):
    return sum(pynvml.nvmlDeviceGetMemoryInfo(h).used
               for h in handles) / (1024 ** 3)

def _total_vram_gb(handles):
    return sum(pynvml.nvmlDeviceGetMemoryInfo(h).total
               for h in handles) / (1024 ** 3)


def run_llm_bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--tp',    type=int, default=1)
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp8'],
                        default='bf16')
    parser.add_argument('--tag',   required=True)
    parser.add_argument('--max_num_seqs', type=int, default=None,
                        help='Cap profile batch size (workaround for large-VRAM GPUs)')
    args = parser.parse_args()

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(args.tp)]

    out_file = 'results/llm_benchmark_results_1.csv'
    if not os.path.exists(out_file):
        with open(out_file, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['Tag', 'Model', 'TP', 'Dtype',
                 'TPS', 'Avg_Lat_ms', 'TTFT_ms', 'VRAM_Util'])

    # ── vLLM engine configuration ─────────────────────────────────────────
    #
    # For bf16/fp16: dtype sets the weight precision, no quantization.
    # For fp8:  quantization="fp8" → W8A8 fp8 (native on H100/B200/B300)
    #
    # Using the quantization= parameter is the only correct way; setting
    # dtype="bfloat16" without quantization= leaves weights in bf16.
    vllm_dtype   = {'bf16': 'bfloat16', 'fp16': 'float16'}.get(args.dtype, 'bfloat16')
    quantization = 'fp8' if args.dtype == 'fp8' else None
    kv_dtype     = 'fp8' if args.dtype == 'fp8' else 'auto'

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=vllm_dtype,
        quantization=quantization,       # None → no quantization applied
        kv_cache_dtype=kv_dtype,
        gpu_memory_utilization=0.85,
    )
    if args.max_num_seqs is not None:
        llm_kwargs['max_num_seqs'] = args.max_num_seqs
    llm = LLM(**llm_kwargs)

    decode_params  = SamplingParams(max_tokens=256, temperature=0.0)
    prefill_params = SamplingParams(max_tokens=1,   temperature=0.0)

    # ── Warmup: two full passes, not timed ───────────────────────────────
    # This ensures CUDA graphs are captured and any JIT compilation is done
    # before any measurement begins.
    llm.generate(PROMPT_POOL[:16], decode_params)
    llm.generate(PROMPT_POOL[:16], decode_params)

    # ── True TTFT: prefill latency for a single request ──────────────────
    # max_tokens=1 means the engine does prefill + exactly 1 decode step.
    # Averaged over 5 repetitions to reduce scheduler jitter.
    ttft_samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        llm.generate([_MEDIUM], prefill_params)
        ttft_samples.append((time.perf_counter() - t0) * 1000)
    ttft_ms = round(sum(ttft_samples) / len(ttft_samples), 2)

    # ── Peak throughput: sweep concurrent-request counts ─────────────────
    # Different GPUs saturate at different concurrencies. We sweep and keep
    # the best (peak) TPS so the result is hardware-limited, not batch-limited.
    best_tps = 0.0
    best_lat = 0.0

    for n_concurrent in [8, 32, 64, 128, 150]:
        batch = PROMPT_POOL[:n_concurrent]
        t0 = time.perf_counter()
        outputs = llm.generate(batch, decode_params)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed
        lat = (elapsed / len(batch)) * 1000
        if tps > best_tps:
            best_tps = tps
            best_lat = lat

    # ── VRAM utilisation (measured, not hardcoded) ────────────────────────
    used_gb  = _used_vram_gb(handles)
    total_gb = _total_vram_gb(handles)
    vram_util = round(used_gb / total_gb, 3)

    with open(out_file, 'a', newline='') as f:
        csv.writer(f).writerow([
            args.tag, args.model, args.tp, args.dtype,
            round(best_tps, 2), round(best_lat, 2), ttft_ms, vram_util,
        ])

    print(f'  {args.tag} {args.dtype}: peak {best_tps:,.0f} TPS | '
          f'TTFT {ttft_ms:.1f} ms | VRAM {vram_util:.3f} '
          f'({used_gb:.1f}/{total_gb:.1f} GB)')

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    run_llm_bench()
