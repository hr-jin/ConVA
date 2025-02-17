#!/bin/bash

VALUES=(
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

TRAIN_DATA_PATH="/home/xxx/Value_FULCRA/data/data_hybrid.jsonl"

ORIGIN_MODEL_PATH="/data/models/llama-2-7b-chat"
BASE_OUTPUT_DIR="/home/xxx/llama-2-7b-chat/lora_results" 
SCRIPT="/home/xxx/GCAV/llama-2-7b-chat_Lora_training.py"


for VALUE in "${VALUES[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${VALUE}"
    LOG_FILE="${OUTPUT_DIR}/training.log"

    mkdir -p "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES="0" nohup python -u $SCRIPT \
        --value "$VALUE" \
        --train_data_path "$TRAIN_DATA_PATH" \
        --origin_model_path "$ORIGIN_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE" 2>&1 &

    echo "value: $VALUE"
    echo "log file: $LOG_FILE"
    echo "process ID: $!"
    echo "----------------------------------------"
done