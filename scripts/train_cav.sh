#!/bin/bash

cd /home/xxx/GCAV

cav_data_name=(
    "achievement_hard"
    "stimulation_hard"
    "hedonism_hard"
    "self-direction_hard"
    "power_hard"
    "security_hard"
    "tradition_hard"
    "conformity_hard"
    "benevolence_hard"
    "universalism_hard"   
)

base_model=(
    # "/data/models/llama-2-7b-chat"
    "/data/models/llama-3-8b-Instruct"
    # "/data/models/Qwen2.5-7B-Instruct"
    # "/data/models/Qwen2.5-14B-Instruct"
)

cd /home/xxx/GCAV
num_train_pairs=100


for base in "${base_model[@]}"; do
    for name in "${cav_data_name[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -u train_cav.py \
            --base_model "$base" \
            --cav_data_name "$name"\
            --num_train_pairs $num_train_pairs
    done
done