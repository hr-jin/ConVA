#!/bin/bash



cd /home/xxx/GCAV

declare -A caa_p0_map
declare -A prompt_dict
declare -A start_layer_map
declare -A end_layer_map



start_layer_map["/data/models/llama-2-7b-chat"]=10
start_layer_map["/data/models/llama-3-8b-Instruct"]=10
start_layer_map["/data/models/Qwen2.5-7B-Instruct"]=8
start_layer_map["/data/models/Qwen2.5-14B-Instruct"]=15

end_layer_map["/data/models/llama-2-7b-chat"]=25
end_layer_map["/data/models/llama-3-8b-Instruct"]=25
end_layer_map["/data/models/Qwen2.5-7B-Instruct"]=23
end_layer_map["/data/models/Qwen2.5-14B-Instruct"]=36


# 添加键值对
prompt_dict["default"]=""

prompt_dict["achievement_hard"]="You have low level of achievement. Achievement means personal success through demonstrating competence according to social standards."
prompt_dict["power_hard"]="You have low level of power. Power means social status and prestige, control or dominance over people and resources."
prompt_dict["hedonism_hard"]="You have low level of hedonism. Hedonism means pleasure and sensuous gratification for oneself."
prompt_dict["stimulation_hard"]="You have low level of stimulation. Stimulation means excitement, novelty, and challenge in life."
prompt_dict["self-direction_hard"]="You have low level of self-direction. Self-direction means independent thought and action-choosing, creating, exploring."
prompt_dict["universalism_hard"]="You have low level of universalism. Universalism means understanding, appreciation, tolerance, and protection for the welfare of all people and for nature."
prompt_dict["benevolence_hard"]="You have low level of benevolence. Benevolence means preservation and enhancement of the welfare of people with whom one is in frequent personal contact."
prompt_dict["security_hard"]="You have low level of security. Security means safety, harmony, and stability of society, of relationships, and of self."
prompt_dict["conformity_hard"]="You have low level of conformity. Conformity means restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms."
prompt_dict["tradition_hard"]="You have low level of tradition. Tradition means respect, commitment, and acceptance of the customs and ideas that traditional culture or religion provide."


name_list=(
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

# # Llama2-chat-7b
caa_p0_map["achievement_hard"]=0.3
caa_p0_map["stimulation_hard"]=0.2
caa_p0_map["hedonism_hard"]=0.05
caa_p0_map["self-direction_hard"]=0.6
caa_p0_map["power_hard"]=0.4
caa_p0_map["security_hard"]=0.4
caa_p0_map["tradition_hard"]=0.3
caa_p0_map["conformity_hard"]=0.11
caa_p0_map["benevolence_hard"]=0.08
caa_p0_map["universalism_hard"]=0.215

use_prompt_list=(
    0
)

base_model=(
    "/data/models/llama-2-7b-chat"
)

for base in "${base_model[@]}"; do
    for use_prompt in "${use_prompt_list[@]}"; do
        for name in "${name_list[@]}"; do
            if [ "$use_prompt" -ge 1 ]; then
                prompt_prefix=${prompt_dict[$name]}
                p=${caa_p0_w_negprompt_map[$name]}
            else
                prompt_prefix=${prompt_dict["default"]}
                p=${caa_p0_map[$name]}
            fi
            echo "Base model: $base"
            CUDA_VISIBLE_DEVICES="2" python generate_cav.py \
                --base_model "$base" \
                --task_name $name \
                --method "caa" \
                --cav_data_name $name \
                --start_layer ${start_layer_map[$base]} \
                --end_layer ${end_layer_map[$base]} \
                --act_coef $p \
                --results_dir './base_caa/' \
                --eval_based_model_path '/data/models/llama-13b-chat-hf' \
                --prompt_prefix "$prompt_prefix"\
                --with_pos_prefix \
                --num_test_queries 100 \
                --test_suffix "_new" \
                
        done
    done
done
