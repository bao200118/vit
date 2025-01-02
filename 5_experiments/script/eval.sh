#!/bin/bash

# Script train
#PC
#PY_PATH="/c/Users/Admin/anaconda3/envs/vit/python.exe"

#Server 
PY_PATH="/home/nguyenvd_drone/miniconda3/envs/andt/bin/python"
MODEL_PATH="/storageStudents/ncsmmlab/tungufm/VitFromScratch/final/vit_v2_final.pth"

#- chạy file eval.py với demo_mod2.py
#- nhớ rằng là do cách lưu model ở file ta train demo_mod2.py ở dạng state dict nên k đủ tham số để visualize, 
  # ta phải filter lại state dict để loại đi các tham số như total_param vì sử dụng để đọc tổng param trước đó
#- update ở demo_mod2.py là return attention_weights, bật mode T/F để visualize

#update mới nhất thử lại với eval3.py
## Truyền tham số của vit_v2  
$PY_PATH eval3.py \
--model $MODEL_PATH \
--img-size 64 \
--batch-size 32 \
--workers 4 \
--name vit_v2_test \
--gpu_id 1 \
--visualize
