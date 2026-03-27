#!/bin/bash
#SBATCH --job-name=gpu_benchmark
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64          
#SBATCH --mem=500G                  
#SBATCH --output=bench_%j.out

# Manual Partition-Specific Setup
# sbatch --partition=a100 --gres=gpu:8 run_bench.sh
# sbatch --partition=a100 --gres=gpu:8 --exclude=c013,c012 run_bench.sh
# sbatch --partition=h100 --gres=gpu:4 run_bench.sh
# sbatch --partition=b200 --gres=gpu:8 run_bench.sh
# sbatch --partition=b300 --gres=gpu:8 run_bench.sh

if [[ "$SLURM_JOB_PARTITION" == "h100" ]]; then
    MAX_GPUS=4
else
    MAX_GPUS=8
fi

module load anaconda3
module load cuda/13.0.2
source activate /weka/scratch/dzarzhi1/ext-zand/py311

# Blackwell/Hopper Performance & Safety Flags
if [[ "$SLURM_JOB_PARTITION" == "b200" || "$SLURM_JOB_PARTITION" == "b300" ]]; then
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    export TRITON_PTXAS_PATH=$(which ptxas)
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

echo "### STARTING BENCHMARK ON ${SLURM_JOB_PARTITION} (${MAX_GPUS} GPUs) ###"

# --- 1. COMPUTER VISION BENCHMARK (1, 4, 8 GPUs) ---
for DTYPE in "bf16" "fp16"; do
    python cv.py --models "resnet50,vit_l_16" --batch_sizes "256,1024" --tag ${SLURM_JOB_PARTITION}_1g --dtype $DTYPE --gpus 1
    python cv.py --models "resnet50,vit_l_16" --batch_sizes "1024,4096" --tag ${SLURM_JOB_PARTITION}_4g --dtype $DTYPE --gpus 4
    if [ "$MAX_GPUS" -eq 8 ]; then
        python cv.py --models "resnet50,vit_l_16" --batch_sizes "4096,16384" --tag ${SLURM_JOB_PARTITION}_8g --dtype $DTYPE --gpus 8
    fi
done



# --- 2. LLM BENCHMARK (1, 4, 8 GPUs) ---
# 1-GPU LLM (8B and 70B - 70B only on B200/B300)
#LLM_DTYPES=("bfloat16" "float16")
LLM_DTYPES=("bf16" "fp16")
if [[ "$SLURM_JOB_PARTITION" != "a100" ]]; then LLM_DTYPES+=("fp8"); fi
if [[ "$SLURM_JOB_PARTITION" == "b200" || "$SLURM_JOB_PARTITION" == "b300" ]]; then LLM_DTYPES+=("fp4"); fi

for DTYPE in "${LLM_DTYPES[@]}"; do
    python llm.py --model models/Llama-3-8B --tp 1 --dtype $DTYPE --tag ${SLURM_JOB_PARTITION}_8b_1g
    if [[ "$SLURM_JOB_PARTITION" == "b200" || "$SLURM_JOB_PARTITION" == "b300" ]]; then
        python llm.py --model models/Llama-3-70B --tp 1 --dtype $DTYPE --tag ${SLURM_JOB_PARTITION}_70b_1g
    fi

    # 4-GPU LLM (8B and 70B)
    python llm.py --model models/Llama-3-8B --tp 4 --dtype $DTYPE --tag ${SLURM_JOB_PARTITION}_8b_4g
    python llm.py --model models/Llama-3-70B --tp 4 --dtype $DTYPE --tag ${SLURM_JOB_PARTITION}_70b_4g

    # 8-GPU LLM (Skip for H100)
    if [ "$MAX_GPUS" -eq 8 ]; then
        python llm.py --model models/Llama-3-8B --tp 8 --dtype $DTYPE --tag ${SLURM_JOB_PARTITION}_8b_8g
        python llm.py --model models/Llama-3-70B --tp 8 --dtype $DTYPE --tag ${SLURM_JOB_PARTITION}_70b_8g
    fi
done

echo "### ${SLURM_JOB_PARTITION} COMPLETE ###"