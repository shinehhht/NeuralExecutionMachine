#!/bin/bash

LR=(0.0001)
BITS_LIST=(5)
ALPHA=(0.95)
VARIANTS=("ema")

DATASET_PATH_PREFIX="multibit_add_"
DATASET_PATH_SUFFIX="_dataset_10k.pt"
EPOCHS=1000
BATCH_SIZE=2048


for lr in "${LR[@]}"; do
  for bits in "${BITS_LIST[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        DATASET_PATH="${DATASET_PATH_PREFIX}${bits}bit${DATASET_PATH_SUFFIX}"
            echo "Dataset path: $DATASET_PATH"

            for alpha in "${ALPHA[@]}"; do
                FILE_NAME="uniform_${alpha}${variant}_add_${bits}bit_10k"
                echo "=== Running: bits=${bits}, variant=${variant} ===, alpha=${alpha}, lr=${lr} ==="

                CMD="torchrun --nproc_per_node=8 \
                    --master_port=29501 \
                    training/tinyTransformer_train_shared.py \
                    --file $FILE_NAME \
                    --dataset_file $DATASET_PATH \
                    --input_bits $bits \
                    --output_bits $((bits+1)) \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --lr $lr \
                    --betas 0.9 0.98 \
                    --weight_decay 1e-5 \
                    --gate_decay 0.99 \
                    --gate_alpha $alpha  \
                    --input_l_bits 34"

                CMD="$CMD --gate_ema"
                eval $CMD
            done
    done
  done
done

