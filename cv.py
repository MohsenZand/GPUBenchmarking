import gc, torch, argparse, pynvml, csv, os, time
import torchvision.models as models

def get_stats(handle):
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pow_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util.gpu, pow_draw, mem_info.used / (1024**3)
    except: return 0, 0, 0

def run_bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, required=True)
    parser.add_argument('--batch_sizes', type=str, required=True)
    parser.add_argument('--dtype', type=str, choices=['bf16', 'fp16'], default='bf16') # Removed fp8 for CV
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    csv_file = "cv_benchmark_results_1.csv"
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("Tag,Model,Dtype,GPUs,Batch,FPS,Lat_ms,Power_W,Img_per_Watt,VRAM_GB\n")

    # Mapping string to torch dtype
    dtype_map = {
                    "bf16": torch.bfloat16, 
                    "fp16": torch.float16, 
                    #"fp8": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
                }
    
    target_dtype = dtype_map[args.dtype]
    device = "cuda"
    
    for m_name in args.models.split(','):
        for bs in [int(x) for x in args.batch_sizes.split(',')]:
            try:
                gc.collect(); torch.cuda.empty_cache()
                
                # Initialize model and move to target dtype immediately
                model = getattr(models, m_name)(weights=None).to(device)
                model = model.to(target_dtype)
                
                # Wrap for Multi-GPU after the cast
                if args.gpus > 1:
                    model = torch.nn.DataParallel(model)
                
                model.eval()
                
                # Create input. randn doesn't support fp8, so create in bf16 then cast
                if args.dtype == "fp8":
                    img = torch.randn(bs, 3, 224, 224, device=device, dtype=torch.bfloat16).to(target_dtype)
                else:
                    img = torch.randn(bs, 3, 224, 224, device=device, dtype=target_dtype)

                # Warmup
                with torch.no_grad():
                    for _ in range(10): _ = model(img)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(50): _ = model(img)
                torch.cuda.synchronize()
                end = time.perf_counter()

                fps = (bs * 50) / (end - start)
                lat = ((end - start) / 50) * 1000
                gpu_util, pwr, vram = get_stats(handle)

                with open(csv_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([args.tag, m_name, args.dtype, args.gpus, bs, round(fps,2), round(lat,2), round(pwr,2), round(fps/pwr,2), round(vram,2)])
                
                del model, img
            except Exception as e:
                print(f"OOM or Error on {m_name} {args.dtype} BS {bs}: {e}")

if __name__ == "__main__":
    run_bench()