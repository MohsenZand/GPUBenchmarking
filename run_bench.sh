#!/bin/bash
#SBATCH --job-name=gpu_benchmark
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --output=logs/bench_schmidt_%j.out
#SBATCH --error=logs/bench_schmidt_%j.err
#
# Submit examples:
#   sbatch --partition=a100 --gres=gpu:8              run_bench.sh
#   sbatch --partition=a100 --gres=gpu:8 --exclude=c013,c012 run_bench.sh
#   sbatch --partition=h100 --gres=gpu:4              run_bench.sh
#   sbatch --partition=b200 --gres=gpu:8              run_bench.sh
#   sbatch --partition=b300 --gres=gpu:8              run_bench.sh

# ── Environment ─────────────────────────────────────────────────────────────


export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_DISABLE_VERSION_CHECK=1   # guards against flashinfer-cubin/python skew

# flashinfer JIT for Blackwell (sm_100a) links against libcuda; conda puts the
# stub in lib/stubs, not lib64/stubs where the linker searches by default.
export LIBRARY_PATH="${CONDA_PREFIX}/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"

nvidia-smi

# ── GPU / tag detection ──────────────────────────────────────────────────────
# After the job starts, nvidia-smi reflects the allocated GPUs.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
if   [[ $GPU_NAME == *"H200"* ]]; then TAG="h200"
elif [[ $GPU_NAME == *"H100"* ]]; then TAG="h100"
elif [[ $GPU_NAME == *"B200"* ]]; then TAG="b200"
elif [[ $GPU_NAME == *"B300"* ]]; then TAG="b300"
else TAG="a100"; fi

MAX_GPUS=$(nvidia-smi -L | wc -l)

# Blackwell (sm_100) only: Triton arch + ptxas path
if [[ $TAG == "b200" || $TAG == "b300" ]]; then
    export TRITON_OVERRIDE_ARCH=sm100
    export PYTORCH_TRITON_ARCH=sm100
    export TRITON_PTXAS_PATH=$(which ptxas)
fi

echo "=== Partition: ${SLURM_JOB_PARTITION} | Tag: ${TAG} | GPUs: ${MAX_GPUS} ==="
echo "=== Node: ${SLURMD_NODENAME} | Job ID: ${SLURM_JOB_ID} ==="

# ── torchrun config (single-node) ───────────────────────────────────────────
# --standalone picks a free port automatically; safe for sequential launches
# and avoids port-conflict issues between concurrent jobs on the same node.
TORCHRUN="torchrun --standalone"

# ── CV Inference  (each rank runs independently; DDP only for all-reduce) ───
CV_INF_MODELS="resnet50,efficientnet_b4,vit_l_16,convnext_base"

for DTYPE in "bf16" "fp16"; do
    $TORCHRUN --nproc_per_node=1 src/cv.py \
        --models "$CV_INF_MODELS" --batch_sizes "64,256,1024" \
        --tag ${TAG}_1g --dtype $DTYPE --mode inference

    if [ "$MAX_GPUS" -ge 4 ]; then
        $TORCHRUN --nproc_per_node=4 src/cv.py \
            --models "$CV_INF_MODELS" --batch_sizes "256,1024,4096" \
            --tag ${TAG}_4g --dtype $DTYPE --mode inference
    fi

    if [ "$MAX_GPUS" -ge 8 ]; then
        $TORCHRUN --nproc_per_node=8 src/cv.py \
            --models "$CV_INF_MODELS" --batch_sizes "1024,4096,8192" \
            --tag ${TAG}_8g --dtype $DTYPE --mode inference
    fi
done

# ── CV Training  (DDP with gradient all-reduce across ranks) ─────────────────
CV_TRAIN_MODELS="resnet50,vit_l_16"

for DTYPE in "bf16" "fp16"; do
    $TORCHRUN --nproc_per_node=1 src/cv.py \
        --models "$CV_TRAIN_MODELS" --batch_sizes "64,256" \
        --tag ${TAG}_1g --dtype $DTYPE --mode train

    if [ "$MAX_GPUS" -ge 4 ]; then
        $TORCHRUN --nproc_per_node=4 src/cv.py \
            --models "$CV_TRAIN_MODELS" --batch_sizes "256,1024" \
            --tag ${TAG}_4g --dtype $DTYPE --mode train
    fi

    if [ "$MAX_GPUS" -ge 8 ]; then
        $TORCHRUN --nproc_per_node=8 src/cv.py \
            --models "$CV_TRAIN_MODELS" --batch_sizes "1024,4096" \
            --tag ${TAG}_8g --dtype $DTYPE --mode train
    fi
done

# ── LLM  (vLLM handles tensor parallelism internally) ────────────────────────
LLM_8B="models/Llama-3-8B"
LLM_70B="models/Llama-3-70B"

LLM_DTYPES=("bf16" "fp16")
if [[ $TAG != "a100" ]]; then LLM_DTYPES+=("fp8"); fi
# fp4 (NV-FP4 / petit_nvfp4) requires pre-quantized model checkpoints not
# available for standard Llama-3 HF weights; skipped for now.

for DTYPE in "${LLM_DTYPES[@]}"; do
    python src/llm.py --model $LLM_8B  --tp 1 --dtype $DTYPE --tag ${TAG}_8b_1g
    if [[ $TAG == "b200" || $TAG == "b300" ]]; then
        python src/llm.py --model $LLM_70B --tp 1 --dtype $DTYPE --tag ${TAG}_70b_1g
    fi

    if [ "$MAX_GPUS" -ge 4 ]; then
        python src/llm.py --model $LLM_8B  --tp 4 --dtype $DTYPE --tag ${TAG}_8b_4g
        python src/llm.py --model $LLM_70B --tp 4 --dtype $DTYPE --tag ${TAG}_70b_4g
    fi

    if [ "$MAX_GPUS" -ge 8 ]; then
        python src/llm.py --model $LLM_8B  --tp 8 --dtype $DTYPE --tag ${TAG}_8b_8g
        python src/llm.py --model $LLM_70B --tp 8 --dtype $DTYPE --tag ${TAG}_70b_8g
    fi
done

echo "=== ${TAG} benchmark complete ==="
