#!/bin/bash

cd /home/xxx/GCAV

declare -A prompt_dict
declare -A cav_p0_map
declare -A cav_p0_w_negprompt_map

prompt_dict["default"]=""

name_list=(
    "achievement"
    "stimulation"
    "hedonism"
    "self-direction"
    "power"
    "security"
    "tradition"
    "conformity"
    "benevolence"
    "universalism"
)

base_model=(
    "/data/models/llama-2-7b-chat"
    # "/data/models/llama-3-8b-Instruct"
    # "/data/models/Qwen2.5-7B-Instruct"
    # "/data/models/Qwen2.5-14B-Instruct"
)

use_prompt_list=(
    0
)

for base in "${base_model[@]}"; do
    for use_prompt in "${use_prompt_list[@]}"; do
        for name in "${name_list[@]}"; do
            if [ "$use_prompt" -ge 1 ]; then
                prompt_prefix=${prompt_dict[$name]}
            else
                prompt_prefix=${prompt_dict["default"]}
            fi

            CUDA_VISIBLE_DEVICES="1" python -u generate_cav.py \
                --base_model "$base" \
                --task_name $name \
                --method "sft" \
                --lora_path "/home/xxx/llama-2-7b-chat/lora_results_continue/$name" \
                --results_dir './baselines/' \
                --eval_based_model_path '/data/models/llama-13b-chat-hf' \
                --prompt_prefix "$prompt_prefix"\
                --with_pos_prefix \
                --test_suffix "_new" \
                --num_test_queries 100 \
                
        done
    done
done