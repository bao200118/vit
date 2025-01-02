#!/bin/bash

# Script để chạy huấn luyện mô hình
# PY_PATH="/c/Users/Admin/anaconda3/envs/vit/python.exe"
PY_PATH="/home/nguyenvd_drone/miniconda3/envs/andt/bin/python"


##### mô hình đã tích hợp gumbel softmax, các data augmentation, hàm Relu thay cho Gelu ở lớp mlp FFN
$PY_PATH demo_vit_gumbel.py \
--name 'vit_v2_gumbel' \
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

