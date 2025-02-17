import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from ..utils import *
from tqdm import tqdm
from ..load_data import load_test_data, add_template_and_prefix
import logging
import warnings

from mmlu import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from peft import PeftModel
import json
from peft import LoraConfig



seed = 42
set_seed(seed)

def one_generate_(
    model: AutoModelForCausalLM, 
    tokenizer, 
    user_prompt, 
    max_new_tokens=100, 
    temperature=0.01, 
    model_name='llama-2', 
    use_stop_texts=True,
    return_logits=False,
):
    if not return_logits:
        inputs           = tokenizer(user_prompt, return_tensors="pt")
        outputs          = model.generate(
            **inputs.to(model.device), 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.eos_token_id, 
            temperature=temperature, 
            top_p=1.0, 
            top_k=0,
            eos_token_id=tokenizer.convert_tokens_to_ids(STOP_TEXTS[0]) if len(STOP_TEXTS) > 0 else tokenizer.eos_token_id,
        )
        decoded_input  = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True).replace('�', '')
        origin_text    = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('�', '')

        
        if ('llama-3' in model_name.lower()) or ('qwen' in model_name.lower()):
            response_text = origin_text[len(decoded_input):].strip()
        else:
            response_text = origin_text[len(user_prompt):].strip()
        if use_stop_texts:
            for stop_text in STOP_TEXTS:
                if stop_text in response_text:
                    response_text = response_text[:-1-len(response_text.split(stop_text)[-1])]
        return response_text
    else:
        inputs = tokenizer(user_prompt, return_tensors="pt")
        outputs = model(
            **inputs.to(model.device), 
        )
        
        logits = outputs.logits[0,-1].tolist()  
        vocab = tokenizer.get_vocab()

        logits_dict = {token: logits[idx] for idx, token in enumerate(list(vocab.keys()))}
        
        logits_dict = {
            'A': logits_dict['A'],
            'B': logits_dict['B'],
            'C': logits_dict['C'],
            'D': logits_dict['D'],
        }
        max_key = max(logits_dict, key=logits_dict.get)


        print(logits_dict, max_key)
        return max_key
        

def generate(
    base_model, 
    task_name, 
    lora_path,
    with_pos_prefix=False,
    prompt_prefix=None,
    return_tag=False,
    max_new_tokens=32,
    num_test_queries=100,
    test_suffix='',
    
):
    model_name = base_model.split('/')[-1]
    ##### Load base model #####
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, cache_dir='/data/huggingface/', local_files_only=True).cuda().eval()
    
    with open(os.path.join(lora_path,"adapter_config.json"), "r") as f:
        config_data = json.load(f)

    lora_config = LoraConfig(**config_data)
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        cache_dir='/data/huggingface/',
        local_files_only=True
    ) 
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token
    print(model.device)

    STOP_TEXTS.append(tokenizer.eos_token)
    
    if with_pos_prefix and prompt_prefix is not None:
        prefix = prompt_prefix
    else:
        prefix = None
        
    _, prompt_list_origin, prompt_list_wo_tplt = load_test_data(
        task_name, 
        return_tag, 
        prefix,  
        model_name=model_name, 
        num_test_queries=num_test_queries,
        test_suffix=test_suffix
    )

    scores = check_human_values(prompts={task_name: prompt_list_wo_tplt}, value=task_name)
    
    input_prompt_list, prompt_list_origin, prompt_list_wo_tplt = load_test_data(
        task_name, 
        return_tag, 
        prefix, 
        model_name=model_name, 
        scores=scores[task_name], 
        threshold=VALUE_THRESHOLDS[task_name],
        low_threshold=VALUE_LOW_THRESHOLDS[task_name],
        num_test_queries=num_test_queries,
        test_suffix=test_suffix
    )
    
    print('Filtered questions ratio:',sum((scores[task_name] < VALUE_THRESHOLDS[task_name]) & (scores[task_name] > VALUE_LOW_THRESHOLDS[task_name])) / scores[task_name].shape[0])
    
    used_prompt_list = []
    generated = []
    control_label_list = []
    for idx, (ques, oris_ques) in tqdm(enumerate(zip(input_prompt_list, prompt_list_origin)), total=len(input_prompt_list)):
        if (scores is not None):
            print('score: {}, threshold: {}, low threshold: {}'.format(scores[task_name][idx], VALUE_THRESHOLDS[task_name], VALUE_LOW_THRESHOLDS[task_name]))
        if (scores is not None) and ((scores[task_name][idx] > VALUE_THRESHOLDS[task_name]) or (scores[task_name][idx] < VALUE_LOW_THRESHOLDS[task_name])):
            control_label_list.append(1)
            gen_str = one_generate_(model, tokenizer, ques, max_new_tokens=max_new_tokens, model_name=model_name)
            used_prompt_list.append(ques)
        else:
            control_label_list.append(0)
            gen_str = one_generate_(model, tokenizer, oris_ques, max_new_tokens=max_new_tokens, model_name=model_name)
            used_prompt_list.append(oris_ques)
        generated.append(gen_str)

    result_df = pd.DataFrame({'prompt':used_prompt_list, 'processed_prompt':prompt_list_wo_tplt, 'generated':generated, 'prompt_list_origin': prompt_list_origin, 'control_label': control_label_list})
        
    return result_df
        
        
def check_human_values(prompts, value):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_path = "/home/xxx/GCAV/Deberta_Human_Value_Detector"
        value_detector_tokenizer =  AutoTokenizer.from_pretrained(model_path, use_fast=False)
        value_detector = AutoModelForSequenceClassification.from_pretrained(model_path,trust_remote_code=True, device_map={'': 'cuda:0'}).eval()

    results = {}
    for task, example_text_list in prompts.items():
        results[task] = {}
        results[task]['texts'] = []
        results[task]['labels'] = []
        results[task]['test_prediction'] = []
        for example_text in tqdm(example_text_list):
            encoding = value_detector_tokenizer(
                example_text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_tensors='pt',
            ).to('cuda:0')
            
            with torch.no_grad():
                test_prediction = value_detector(encoding["input_ids"], encoding["attention_mask"])
                test_prediction = test_prediction["output"].flatten().cpu().numpy()
        
            results[task]['texts'].append(example_text)
            results[task]['labels'].append(LABEL_COLUMNS)
            results[task]['test_prediction'].append(test_prediction)
        
    test_columns = []
    for label in LABEL_COLUMNS:
        if value in label.lower():
            test_columns.append(label)
            
    final_results = {}

    for idx in range(len(test_columns)):
        dim = test_columns[idx]
        print('dim:', dim)
        task_to_check = dim.split(':')[0].lower()

        for l_idx, task in enumerate(LABEL_COLUMNS):
            if task == dim:
                label_idx = l_idx
                print('label_idx:',label_idx)

        if task_to_check not in final_results: 
            final_results[task_to_check] = []
                    
        scores = np.array(results[task_to_check]['test_prediction'])[:,label_idx]
        final_results[task_to_check].append(scores)

    for key in final_results.keys():
        final_results[key] = np.array(final_results[key]).mean(0)
            
    torch.cuda.empty_cache()
    return final_results


def check_human_values_mmlu(prompts, value):
    model_path = "/home/xxx/GCAV/Deberta_Human_Value_Detector"
    value_detector_tokenizer =  AutoTokenizer.from_pretrained(model_path, use_fast=False)
    value_detector = AutoModelForSequenceClassification.from_pretrained(model_path,trust_remote_code=True, device_map={'': 'cuda:0'}).eval()

    results = {}
    for task, example_text_list in prompts.items():
        results[task] = {}
        results[task]['texts'] = []
        results[task]['labels'] = []
        results[task]['test_prediction'] = []
        for example_text in tqdm(example_text_list):
            encoding = value_detector_tokenizer(
                example_text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_tensors='pt',
            ).to('cuda:0')
            
            with torch.no_grad():
                test_prediction = value_detector(encoding["input_ids"], encoding["attention_mask"])
                test_prediction = test_prediction["output"].flatten().cpu().numpy()
        
            results[task]['texts'].append(example_text)
            results[task]['labels'].append(LABEL_COLUMNS)
            results[task]['test_prediction'].append(test_prediction)
        
    test_columns = []
    for label in LABEL_COLUMNS:
        if value in label.lower():
            test_columns.append(label)
            
    final_results = {}
    for task, example_text_list in prompts.items():
        final_results[task] = {}

    task_list_to_check = []
    for idx in range(len(test_columns)):
        dim = test_columns[idx]
        print('dim:', dim)
        task_to_check = dim.split(':')[0].lower()
        task_list_to_check.append(task_to_check)

        for l_idx, task in enumerate(LABEL_COLUMNS):
            if task == dim:
                label_idx = l_idx
                print('label_idx:',label_idx)

        for key in prompts.keys():
            if task_to_check not in final_results[key]: 
                final_results[key][task_to_check] = []
                    
        for task, example_text_list in prompts.items():
            scores = np.array(results[task]['test_prediction'])[:,label_idx]
            final_results[task][task_to_check].append(scores)
        
    for task, example_text_list in prompts.items():
        for key in set(task_list_to_check):
            final_results[task][key] = np.array(final_results[task][key]).mean(0)
            
    torch.cuda.empty_cache()
    return final_results