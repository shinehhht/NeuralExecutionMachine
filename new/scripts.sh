
#python training/mlp_test_train.py --mode gradbit   --file benefit_gradbit_tiny_group30——actual   --group_num 30 --epochs 50


python training/tinyTransformer_train.py   --file grad_norm_difflr  --epochs 300 --lr 1e-4 --betas 0.9 0.98 --weight_decay 1e-5

#python training/tinyTransformer_train.py   --file add_3bit_10k_  --epochs 300 --lr 1e-5 --betas 0.9 0.97 --weight_decay 1e-3

#python training/tinyTransformer_train.py   --file add_3bit_10k_  --epochs 300 --lr 5e-6 --betas 0.9 0.999 --weight_decay 5e-6



