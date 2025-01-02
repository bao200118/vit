#!/bin/bash

# Script để chạy huấn luyện mô hình
PY_PATH="/c/Users/Admin/anaconda3/envs/vit/python.exe"

# # Run the demo.py script with default arguments
# $PY_PATH demo3.py \
# --name 'vit_v1' \
# --img_size 32 \
# --depth 12 \
# --heads 8 \
# --patch_size 4 \
# --hidden_dim 256 \
# --mlp_size 512 \
# --dropout_rate 0.1 \
# --batch_sz 32 \
# --epochs 50 \
# --learning_rate 0.003 \
# --gpu_id 0 \
# --max_memory_fraction 0.5
# chmod +x demo.sh

# $PY_PATH demo4.py \
# --name 'vit_v1'
# --img_size 72 \
# --patch_size 6 \
# --hidden_dim 64 \
# --mlp_size 128 \
# --depth 8 \
# --heads 4 \
# --learning_rate 0.001 \
# --batch_sz 64 \
# --epochs 50 \
# --gpu_id 0 \
# --max_memory_fraction 0.5

##### mô hình đã tích hợp gumbel softmax, các data augmentation, hàm Relu thay cho Gelu ở lớp mlp FFN
$PY_PATH demo_mod.py \
--name 'vit_v2' \
--img_size 64 \
--depth 6 \
--heads 4 \
--patch_size 8 \
--hidden_dim 128 \
--mlp_size 256 \
--dropout_rate 0.1 \
--batch_sz 64 \
--epochs 100 \
--learning_rate 0.001 \
--gpu_id 0 \
--max_memory_fraction 0.5

