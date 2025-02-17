from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import argparse
import os

parser = argparse.ArgumentParser(description='Train a model with LoRA.')
parser.add_argument('--value', type=str, required=True, help='The value type to filter the data.')
parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data JSONL file.')
parser.add_argument('--origin_model_path', type=str, required=True, help='Path to the original model.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model.')

args = parser.parse_args()

value = args.value
train_data_path = args.train_data_path
origin_model_path = args.origin_model_path
output_dir = args.output_dir

print("args:", args)


df = pd.read_json(train_data_path, lines=True)
df = df[df['value_types'].apply(lambda x: (f'{value}: +1' in x) or (f'{value}: =1' in x) or (f'{value}: 1' in x) or (f'{value}: +' in x) or (f'{value}: 1+' in x))]


split_result = df['dialogue'].str.split('Bob:', expand=True)

df['instruction'] = split_result[0].str.replace('Human:', '').str.strip()  
df['output'] = split_result[1].str.strip()  

df = df.drop(['value_items', 'query_source', 'value_types', 'dialogue', 'response_source'], axis=1) 

ds = Dataset.from_pandas(df)
print("len(ds):", len(ds))

print("df[:5].values:",df[:5].values)

tokenizer = AutoTokenizer.from_pretrained(origin_model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

max_length = 0
def process_func(example):
    MAX_LENGTH = 600    
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  

    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "len": len(input_ids)
    }
    
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
print("tokenized_id.num_rows:", tokenized_id.num_rows)
print("max_seq_len:", max(tokenized_id['len']))


tokenized_id['len']

import torch

model = AutoModelForCausalLM.from_pretrained(origin_model_path, device_map='cuda',torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

# Lora
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8, 
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)

print("Lora config:", config)
print("model.print_trainable_parameters():", model.print_trainable_parameters())

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=300,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

peft_model_id=output_dir
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)