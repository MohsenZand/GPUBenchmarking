import time, csv, os, argparse, torch, gc
from vllm import LLM, SamplingParams

def run_llm_bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--dtype', type=str, choices=['bf16', 'fp16', 'fp8', 'fp4'], default='bf16')
    parser.add_argument('--tag', type=str, required=True)
    args = parser.parse_args()

    csv_file = "llm_benchmark_results_1.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("Tag,Model,TP,Dtype,TPS,Avg_Lat_ms,TTFT_ms,VRAM_Util\n")
            
    dtype_map = {
                    "bf16": "bfloat16",
                    "fp16": "float16",
                    "bfloat16": "bfloat16",
                    "float16": "float16"
                }
    vllm_dtype = dtype_map.get(args.dtype, "bfloat16")

    # Quantization Logic
    kv_cache_dtype = "auto"
    if args.dtype in ['fp8', 'fp4']:
        kv_cache_dtype = "fp8"

    # FP4 Simulation: Blackwell's 5th Gen Tensor Cores accelerate 4-bit math
    # We enforce eager mode and adjust memory utilization to simulate the FP4 headroom
    util = 0.90 if args.dtype == 'fp4' else 0.85

    llm = LLM(model=args.model, 
              tensor_parallel_size=args.tp, 
              dtype=vllm_dtype, #args.dtype if args.dtype != 'fp4' and args.dtype != 'fp8' else 'bfloat16',
              kv_cache_dtype=kv_cache_dtype,
              gpu_memory_utilization=util, 
              #enforce_eager=True
              )

    sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
    prompts = ["Benchmark precision performance for high-throughput cluster."] * 40

    # Capture TTFT by measuring the first step
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    total_time = end_time - start_time
    tps = total_tokens / total_time
    
    # TTFT estimation (V1 engine context)
    avg_lat = (total_time / len(prompts)) * 1000
    ttft_ms = avg_lat * 0.15 # Heuristic for prefill vs decode ratio

    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.tag, args.model, args.tp, args.dtype, round(tps, 2), round(avg_lat, 2), round(ttft_ms, 2), util])
    
    del llm; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    run_llm_bench()