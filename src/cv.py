"""
CV Benchmark — inference and training modes.

Launch via torchrun (works for 1 or N GPUs):
    torchrun --nproc_per_node=N cv.py --mode inference ...
    torchrun --nproc_per_node=N cv.py --mode train    ...

Outputs:
    cv_benchmark_results_1.csv       (inference)
    cv_train_benchmark_results_1.csv (training)
"""
import gc, os, time, csv, threading, argparse
import torch
import torch.distributed as dist
import torchvision.models as tvmodels
import pynvml

# ── Distributed context set by torchrun ────────────────────────────────────
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))


def setup_dist():
    torch.cuda.set_device(LOCAL_RANK)
    if WORLD_SIZE > 1:
        dist.init_process_group(backend='nccl')


def teardown_dist():
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ── Continuous power sampling (background thread, rank 0 only) ────────────
def _power_sampler(handle, stop_event, readings):
    while not stop_event.is_set():
        try:
            readings.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        except Exception:
            pass
        time.sleep(0.05)


# ── Model registry ─────────────────────────────────────────────────────────
MODEL_BUILDERS = {
    'resnet50':        tvmodels.resnet50,
    'efficientnet_b4': tvmodels.efficientnet_b4,
    'convnext_base':   tvmodels.convnext_base,
    'vit_l_16':        tvmodels.vit_l_16,
}

DTYPE_MAP = {'bf16': torch.bfloat16, 'fp16': torch.float16}

# Inference: more warmup to cover first-call kernel specialisation
INF_WARMUP = 30
INF_ITERS  = 100

# Training: gradient graph builds on warmup, then steady-state is measured
TRAIN_WARMUP = 15
TRAIN_ITERS  = 50


def run_bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',      required=True,
                        help='Comma-separated model names')
    parser.add_argument('--batch_sizes', required=True,
                        help='Comma-separated batch sizes (per-GPU for inference, '
                             'global for training)')
    parser.add_argument('--dtype',   choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--tag',     required=True)
    parser.add_argument('--mode',    choices=['inference', 'train'],
                        default='inference')
    args = parser.parse_args()

    setup_dist()
    device   = torch.device(f'cuda:{LOCAL_RANK}')
    is_main  = (LOCAL_RANK == 0)
    tdtype   = DTYPE_MAP[args.dtype]

    if is_main:
        pynvml.nvmlInit()
        nvml_hdl = pynvml.nvmlDeviceGetHandleByIndex(LOCAL_RANK)
        out_file = ('results/cv_benchmark_results_1.csv'
                    if args.mode == 'inference'
                    else 'results/cv_train_benchmark_results_1.csv')
        if not os.path.exists(out_file):
            with open(out_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'Tag', 'Model', 'Dtype', 'GPUs', 'Batch',
                    'FPS', 'Lat_ms', 'Power_W', 'Img_per_Watt', 'VRAM_GB'
                ])

    for m_name in args.models.split(','):
        builder = MODEL_BUILDERS.get(m_name)
        if builder is None:
            if is_main:
                print(f'  Unknown model "{m_name}", skipping.')
            continue

        for bs in [int(x) for x in args.batch_sizes.split(',')]:
            try:
                gc.collect()
                torch.cuda.empty_cache()

                model = builder(weights=None).to(device).to(tdtype)

                # ── Inference mode ────────────────────────────────────────
                if args.mode == 'inference':
                    model.eval()
                    # bs = per-GPU batch size; total system throughput = bs * WORLD_SIZE
                    img = torch.randn(bs, 3, 224, 224, device=device, dtype=tdtype)

                    with torch.no_grad():
                        for _ in range(INF_WARMUP):
                            model(img)
                    torch.cuda.synchronize()

                    pwr_readings = []
                    stop_evt = threading.Event()
                    if is_main:
                        t = threading.Thread(
                            target=_power_sampler,
                            args=(nvml_hdl, stop_evt, pwr_readings),
                            daemon=True)
                        t.start()

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        for _ in range(INF_ITERS):
                            model(img)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0

                    if is_main:
                        stop_evt.set()
                        t.join()

                    fps = (bs * WORLD_SIZE * INF_ITERS) / elapsed
                    lat = (elapsed / INF_ITERS) * 1000

                # ── Training mode ─────────────────────────────────────────
                else:
                    # Wrap with DDP so gradients are all-reduced across ranks
                    if WORLD_SIZE > 1:
                        model = torch.nn.parallel.DistributedDataParallel(
                            model, device_ids=[LOCAL_RANK])

                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    criterion = torch.nn.CrossEntropyLoss()

                    # bs = global batch size → each rank gets bs // WORLD_SIZE
                    per_gpu_bs = max(1, bs // WORLD_SIZE)
                    img    = torch.randn(per_gpu_bs, 3, 224, 224,
                                        device=device, dtype=tdtype)
                    labels = torch.randint(0, 1000, (per_gpu_bs,), device=device)

                    for _ in range(TRAIN_WARMUP):
                        optimizer.zero_grad()
                        criterion(model(img).float(), labels).backward()
                        optimizer.step()
                    torch.cuda.synchronize()

                    pwr_readings = []
                    stop_evt = threading.Event()
                    if is_main:
                        t = threading.Thread(
                            target=_power_sampler,
                            args=(nvml_hdl, stop_evt, pwr_readings),
                            daemon=True)
                        t.start()

                    t0 = time.perf_counter()
                    for _ in range(TRAIN_ITERS):
                        optimizer.zero_grad()
                        criterion(model(img).float(), labels).backward()
                        optimizer.step()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0

                    if is_main:
                        stop_evt.set()
                        t.join()

                    # Report in global samples/sec (bs is already global)
                    fps = (bs * TRAIN_ITERS) / elapsed
                    lat = (elapsed / TRAIN_ITERS) * 1000

                # ── Record (rank 0 only) ──────────────────────────────────
                if is_main:
                    mem  = pynvml.nvmlDeviceGetMemoryInfo(nvml_hdl)
                    vram = mem.used / (1024 ** 3)
                    pwr  = (sum(pwr_readings) / len(pwr_readings)
                            if pwr_readings else 0.0)

                    with open(out_file, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            args.tag, m_name, args.dtype, WORLD_SIZE, bs,
                            round(fps, 2), round(lat, 2), round(pwr, 2),
                            round(fps / pwr if pwr > 0 else 0.0, 2),
                            round(vram, 2),
                        ])
                    print(f'  [{args.mode}] {m_name} {args.dtype} '
                          f'bs={bs} gpus={WORLD_SIZE}: {fps:,.0f} FPS/sps')

                del model, img

            except Exception as e:
                if is_main:
                    print(f'  ERROR {m_name} {args.dtype} bs={bs}: {e}')

    teardown_dist()


if __name__ == '__main__':
    run_bench()
