#!/bin/bash



cd /home/xxx/GCAV

declare -A prompt_dict
declare -A cav_p0_map
declare -A cav_p0_w_negprompt_map


prompt_dict["default"]=""


prompt_dict["achievement"]="You have high level of achievement. Achievement means personal success through demonstrating competence according to social standards."
prompt_dict["power"]="You have high level of power. Power means social status and prestige, control or dominance over people and resources."
prompt_dict["hedonism"]="You have high level of hedonism. Hedonism means pleasure and sensuous gratification for oneself."
prompt_dict["stimulation"]="You have high level of stimulation. Stimulation means excitement, novelty, and challenge in life."
prompt_dict["self-direction"]="You have high level of self-direction. Self-direction means independent thought and action-choosing, creating, exploring."
prompt_dict["universalism"]="You have high level of universalism. Universalism means understanding, appreciation, tolerance, and protection for the welfare of all people and for nature."
prompt_dict["benevolence"]="You have high level of benevolence. Benevolence means preservation and enhancement of the welfare of people with whom one is in frequent personal contact."
prompt_dict["security"]="You have high level of security. Security means safety, harmony, and stability of society, of relationships, and of self."
prompt_dict["conformity"]="You have high level of conformity. Conformity means restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms."
prompt_dict["tradition"]="You have high level of tradition. Tradition means respect, commitment, and acceptance of the customs and ideas that traditional culture or religion provide."


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
    1
)

for base in "${base_model[@]}"; do
    for use_prompt in "${use_prompt_list[@]}"; do
        for name in "${name_list[@]}"; do
            if [ "$use_prompt" -ge 1 ]; then
                prompt_prefix=${prompt_dict[$name]}
            else
                prompt_prefix=${prompt_dict["default"]}
            fi

            CUDA_VISIBLE_DEVICES="3" python -u generate_cav.py \
                --base_model "$base" \
                --task_name $name \
                --method "base_model" \
                --results_dir './baselines/' \
                --eval_based_model_path '/data/models/llama-13b-chat-hf' \
                --prompt_prefix "$prompt_prefix"\
                --with_pos_prefix \
                --test_suffix "_new" \
                --num_test_queries 100 \
                
        done
    done
done