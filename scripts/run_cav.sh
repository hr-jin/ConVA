#!/bin/bash


cd /home/xxx/GCAV

declare -A prompt_dict
declare -A cav_p0_map
declare -A cav_p0_w_prompt_map
declare -A start_layer_map
declare -A end_layer_map

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


start_layer_map["/data/models/llama-2-7b-chat"]=10
start_layer_map["/data/models/llama-3-8b-Instruct"]=10
start_layer_map["/data/models/Qwen2.5-7B-Instruct"]=8
start_layer_map["/data/models/Qwen2.5-14B-Instruct"]=15

end_layer_map["/data/models/llama-2-7b-chat"]=25
end_layer_map["/data/models/llama-3-8b-Instruct"]=25
end_layer_map["/data/models/Qwen2.5-7B-Instruct"]=23
end_layer_map["/data/models/Qwen2.5-14B-Instruct"]=36

# # w. prompt llama2-7b-chat
cav_p0_w_prompt_map["achievement_hard"]=0.93
cav_p0_w_prompt_map["stimulation_hard"]=0.91
cav_p0_w_prompt_map["hedonism_hard"]=0.83
cav_p0_w_prompt_map["self-direction_hard"]=0.91
cav_p0_w_prompt_map["power_hard"]=0.9
cav_p0_w_prompt_map["security_hard"]=0.9
cav_p0_w_prompt_map["tradition_hard"]=0.92
cav_p0_w_prompt_map["conformity_hard"]=0.7
cav_p0_w_prompt_map["benevolence_hard"]=0.85
cav_p0_w_prompt_map["universalism_hard"]=0.95

# # w.o. prompt llama2-7b-chat
cav_p0_map["achievement_hard"]=0.97
cav_p0_map["stimulation_hard"]=0.93
cav_p0_map["hedonism_hard"]=0.88
cav_p0_map["self-direction_hard"]=0.95
cav_p0_map["power_hard"]=0.9
cav_p0_map["security_hard"]=0.975
cav_p0_map["tradition_hard"]=0.92
cav_p0_map["conformity_hard"]=0.88
cav_p0_map["benevolence_hard"]=0.91
cav_p0_map["universalism_hard"]=0.92

use_prompt_list=(
    1
)


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

base_model=(
    "/data/models/llama-2-7b-chat"
    # "/data/models/llama-3-8b-Instruct"
    # "/data/models/Qwen2.5-14B-Instruct"
)




for base in "${base_model[@]}"; do
    for use_prompt in "${use_prompt_list[@]}"; do
        for name in "${name_list[@]}"; do
            if [ "$use_prompt" -ge 1 ]; then
                prompt_prefix=${prompt_dict[$name]}
                p=${cav_p0_w_prompt_map[$name]}
            else
                prompt_prefix=${prompt_dict["default"]}
                p=${cav_p0_map[$name]}
            fi
            
            CUDA_VISIBLE_DEVICES="1" python -u generate_cav.py \
                --base_model "$base" \
                --task_name $name \
                --method 'cav' \
                --cav_data_name $name \
                --cav_p_0 $p \
                --start_layer ${start_layer_map[$base]} \
                --end_layer ${end_layer_map[$base]} \
                --results_dir './results/' \
                --eval_based_model_path '/data2/data/models/llama-13b-chat-hf' \
                --cav_steer 'plus'\
                --prompt_prefix "$prompt_prefix"\
                --with_pos_prefix \
                --num_test_queries 100 \
                --use_gate 1 \
                --test_suffix "_new" \

        done
    done
done