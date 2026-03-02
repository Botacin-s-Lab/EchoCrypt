#!/usr/bin/env bash
# Few-Shot ASCA Experiments
# Usage: cd fewshot; bash run_experiments.sh

set -euo pipefail

MODEL="ast"
SEEDS=(42 123 456)
K_SHOTS=(1 2 4 8)

declare -A DATASET_MAP=(
    ["new_dataset_phone"]="../new_dataset_phone"
    ["new_dataset_zoom"]="../new_dataset_zoom"
)

echo "============================================"
echo "  Few-Shot ASCA Experiments (AST)"
echo "============================================"

for DATASET in "${!DATASET_MAP[@]}"; do
    DATASET_PATH="${DATASET_MAP[$DATASET]}"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "[WARN] Dataset not found: $DATASET_PATH, skipping."
        continue
    fi

    for K in "${K_SHOTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo ">>> Baseline | dataset=$DATASET k=$K seed=$SEED"
            uv run baseline.py --dataset_dir "$DATASET_PATH" --model "$MODEL" --k_shot "$K" --seed "$SEED" --output_dir results

            echo ""
            echo ">>> ProtoNet (frozen) | dataset=$DATASET k=$K seed=$SEED"
            uv run prototypical.py --dataset_dir "$DATASET_PATH" --model "$MODEL" --k_shot "$K" --seed "$SEED" --mode frozen --output_dir results

            echo ""
            echo ">>> ProtoNet (finetune) | dataset=$DATASET k=$K seed=$SEED"
            uv run prototypical.py --dataset_dir "$DATASET_PATH" --model "$MODEL" --k_shot "$K" --seed "$SEED" --mode finetune --finetune_episodes 2500 --output_dir results
        done
    done
done

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results in fewshot/results/"
echo "============================================"
