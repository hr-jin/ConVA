import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import *
from tqdm import tqdm
from src.load_data import load_test_data
import logging
import warnings

from src.modelwrapper import ModelWrapper
from src.cav import get_reps
from src.load_data import load_cav_train_data
from transformers import AutoModelForSequenceClassification, AutoTokenizer


seed = 42
set_seed(seed)

    

def get_diff_acts(model: ModelWrapper, tokenizer, start_layer, end_layer, pos_prompts=["Love"], neg_prompts=["Hate"]):
    ##### Activate hooks #####

    layer_list = []
    for layer_id in range(start_layer, end_layer + 1, 1):
        layer_name = 'model.layers.{}'.format(str(layer_id))
        layer_list.append(layer_name)
    model.register_forward_hooks(layer_list)

    positive_act = get_reps(model=model, tokenizer=tokenizer, input_examples=pos_prompts, layers=layer_list)   # [1, ...4096]
    negative_act = get_reps(model=model, tokenizer=tokenizer, input_examples=neg_prompts, layers=layer_list)

    diff_acts = {}
    for layer_name in positive_act.keys():
        diff_acts[layer_name] = positive_act[layer_name] - negative_act[layer_name]
    
    
    model.remove_hooks()
    
    return diff_acts    
    



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
                    response_text = response_text.split(stop_text)[0]
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
    cav_train_data_name,
    act_coef:float,
    start_layer:int,
    end_layer:int,
    max_new_tokens=32,
    with_pos_prefix=False, 
    prompt_prefix=None,
    num_test_queries=100,
    test_suffix=''
): 
    
    ##### Load model and tokenizer #####
    model_name = base_model.split('/')[-1]
    hf_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map={'': 'cuda:0'}).cuda().eval()
    model = ModelWrapper(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
    ) 
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token

    STOP_TEXTS.append(tokenizer.eos_token)
    
    pos_prompts, neg_prompts = load_cav_train_data(cav_train_data_name, model_name)
    print('cav_train_data_name:',cav_train_data_name)
    [print(s) for s in pos_prompts[:5]]
    [print(s) for s in neg_prompts[:5]]
    print(len(pos_prompts))
    
    ##### Get model activation hooks #####
    diff_acts = {}
    _diff_acts = get_diff_acts(model, tokenizer, start_layer, end_layer, pos_prompts, neg_prompts) 
    for key in _diff_acts.keys():
        diff_acts[key] = np.mean(_diff_acts[key],axis=0)
    
    steer_dict = {}
    hook_layers = []
    
    for layer_id in range(start_layer, end_layer+1, 1):
        layer_name = 'model.layers.{}'.format(str(layer_id))
        hook_layers.append(layer_name)
        steer_dict[layer_name] = torch.tensor(diff_acts[layer_name]).to(model.device)
    
    model.register_forward_hooks(hook_layers)

    ##### Set control by diff_acts #####
    steer_vecs = {}
    for (layer_name, diff_act) in steer_dict.items():
        steer_vecs[layer_name] = act_coef * diff_act
    
    ##### Load data #####    
    if with_pos_prefix and (prompt_prefix is not None):
        prefix = prompt_prefix
    else:
        prefix = None
        
    ##### Load data #####
    ## get scores
    _, prompt_list_origin, prompt_list_wo_tplt = load_test_data(
        task_name, 
        prefix=prefix, 
        model_name=model_name, 
        num_test_queries=num_test_queries,
        test_suffix=test_suffix
    )
    
    if '_' in cav_train_data_name:
        cav_train_data_name = cav_train_data_name.split('_')[0]
        task_name = task_name.split('_')[0]
    scores = check_human_values(prompts={cav_train_data_name: prompt_list_wo_tplt}, value=cav_train_data_name)
    
    ## get prompts using scores
    prompt_list, prompt_list_origin, prompt_list_wo_tplt = load_test_data(
        task_name, 
        prefix=prefix, 
        model_name=model_name, 
        scores=scores[cav_train_data_name], 
        threshold=VALUE_THRESHOLDS[cav_train_data_name],
        low_threshold=VALUE_LOW_THRESHOLDS[cav_train_data_name],
        num_test_queries=num_test_queries,
        test_suffix=test_suffix
    )
    
    print('Filtered questions ratio:',sum((scores[cav_train_data_name] < VALUE_THRESHOLDS[cav_train_data_name]) & (scores[cav_train_data_name] > VALUE_LOW_THRESHOLDS[cav_train_data_name])) / scores[cav_train_data_name].shape[0])

    used_prompt_list = []
    generated = []
    control_label_list = []
    for idx, (ques, oris_ques) in tqdm(enumerate(zip(prompt_list, prompt_list_origin)), total=len(prompt_list)):
        if (scores is not None):
            print('score: {}, threshold: {}, low threshold: {}'.format(scores[task_name][idx], VALUE_THRESHOLDS[task_name], VALUE_LOW_THRESHOLDS[task_name]))
        if (scores is not None) and ((scores[task_name][idx] > VALUE_THRESHOLDS[task_name]) or (scores[task_name][idx] < VALUE_LOW_THRESHOLDS[task_name])):
            control_label_list.append(1)
            model.clear_cavs()
            model.set_cavs(steer_vecs)
            gen_str = one_generate_(model, tokenizer, ques, max_new_tokens=max_new_tokens, model_name=model_name)
            used_prompt_list.append(ques)
        else:
            control_label_list.append(0)
            model.clear_cavs()
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