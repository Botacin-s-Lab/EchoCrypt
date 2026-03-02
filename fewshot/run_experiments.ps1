# Few-Shot ASCA Experiments
# Usage: cd fewshot; powershell -ExecutionPolicy Bypass -File run_experiments.ps1

$MODEL = "ast"
$SEEDS = @(42, 123, 456)
$K_SHOTS = @(1, 2, 4, 8)

# AST uses raw audio datasets (not spectrogram images)
$DATASET_MAP = @{
    "new_dataset_phone" = "../new_dataset_phone"
    "new_dataset_zoom"  = "../new_dataset_zoom"
}

Write-Host "============================================"
Write-Host "  Few-Shot ASCA Experiments (AST)"
Write-Host "============================================"

foreach ($DATASET in $DATASET_MAP.Keys) {
    $DATASET_PATH = $DATASET_MAP[$DATASET]

    if (-not (Test-Path $DATASET_PATH)) {
        Write-Host "[WARN] Dataset not found: $DATASET_PATH, skipping."
        continue
    }

    foreach ($K in $K_SHOTS) {
        foreach ($SEED in $SEEDS) {
            Write-Host ""
            Write-Host ">>> Baseline | dataset=$DATASET k=$K seed=$SEED"
            python baseline.py --dataset_dir $DATASET_PATH --model $MODEL --k_shot $K --seed $SEED --output_dir results

            Write-Host ""
            Write-Host ">>> ProtoNet (frozen) | dataset=$DATASET k=$K seed=$SEED"
            python prototypical.py --dataset_dir $DATASET_PATH --model $MODEL --k_shot $K --seed $SEED --mode frozen --output_dir results

            Write-Host ""
            Write-Host ">>> ProtoNet (finetune) | dataset=$DATASET k=$K seed=$SEED"
            python prototypical.py --dataset_dir $DATASET_PATH --model $MODEL --k_shot $K --seed $SEED --mode finetune --finetune_episodes 2500 --output_dir results
        }
    }
}

Write-Host ""
Write-Host "============================================"
Write-Host "  All experiments complete!"
Write-Host "  Results in fewshot/results/"
Write-Host "============================================"
