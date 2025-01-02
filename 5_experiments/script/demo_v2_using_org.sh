#!/bin/bash

# Script để chạy huấn luyện mô hình
PY_PATH="/c/Users/Admin/anaconda3/envs/vit/python.exe"

# Truyền tham số
$PY_PATH demo_mod.py \
--name 'vit_v2_using_org' \
--img_size 32 \
--depth 12 \
--heads 12 \
--patch_size 16 \
--hidden_dim 768 \
--mlp_size 3072 \
--dropout_rate 0.1 \
--batch_sz 64 \
--epochs 100 \
--learning_rate 0.001 \
--gpu_id 0 \
--max_memory_fraction 0.5