export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

FILE_NAME="multigpu_add_3bit_10k_"
EPOCHS=1000
BATCH_SIZE=2048

torchrun --nproc_per_node=8 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.98 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-2 \
         --L 0.1 \
         --U2 0.8 \

torchrun --nproc_per_node=8 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.98 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-2 \
         --L 0.1 \
         --U2 0.8 \
         --constrain



torchrun --nproc_per_node=8 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-5 \
         --betas 0.9 0.98 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-2 \
         --L 0.1 \
         --U2 0.8


torchrun --nproc_per_node=8 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-5 \
         --betas 0.9 0.98 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-2 \
         --L 0.1 \
         --U2 0.8 \
         --constrain


torchrun --nproc_per_node=8 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-4 \
         --betas 0.9 0.98 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-2 \
         --L 0.1 \
         --U2 0.8


torchrun --nproc_per_node=8 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-4 \
         --betas 0.9 0.98 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-2 \
         --L 0.1 \
         --U2 0.8 \
         --constrain


torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.999 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-4 \
         --L 1e-2 \
         --U2 0.8

torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.999 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-4 \
         --L 1e-2 \
         --U2 0.8 \
         --constrain


torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.95 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-4 \
         --L 1e-2 \
         --U2 0.8

torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.95 \
         --weight_decay 1e-5 \
         --T1 30 \
         --T2 400 \
         --U1 1e-4 \
         --L 1e-2 \
         --U2 0.8 \
         --constrain


torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.999 \
         --weight_decay 1e-5 \
         --T1 50 \
         --T2 400 \
         --U1 1e-2 \
         --L 1e-2 \
         --U2 0.5 \
         --constrain

torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.999 \
         --weight_decay 1e-5 \
         --T1 50 \
         --T2 400 \
         --U1 1e-2 \
         --L 1e-2 \
         --U2 0.5 \
         --constrain

torchrun --nproc_per_node=4 \
         --master_port=29501 \
         training/tinyTransformer_train_multigpu.py \
         --file $FILE_NAME \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --lr 1e-3 \
         --betas 0.9 0.999 \
         --weight_decay 1e-5 \
         --T1 50 \
         --T2 400 \
         --U1 1e-1 \
         --L 1e-1 \
         --U2 0.6 \
         --constrain


