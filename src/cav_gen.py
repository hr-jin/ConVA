import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modelwrapper import ModelWrapper
import warnings

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

import pickle
from src.cav import get_reps
from src.utils import *
from tqdm import tqdm

import logging

from src.load_data import load_test_data, add_template_and_prefix

from mmlu import MMLU

seed = 42
set_seed(seed)

VALUE_THRESHOLDS_default = {
    "achievement": 0.,
    "stimulation": 0.,
    "hedonism": 0.,
    "self-direction": 0.,
    "power": 0.,
    "security": 0.,
    "tradition": 0.,
    "conformity": 0.,
    "benevolence": 0.,
    "universalism": 0.,
}
VALUE_LOW_THRESHOLDS_default = {
    "achievement": 0.,
    "stimulation": 0.,
    "hedonism": 0.,
    "self-direction": 0.,
    "power": 0.,
    "security": 0.,
    "tradition": 0.,
    "conformity": 0.,
    "benevolence": 0.,
    "universalism": 0.,
}

def generate_(
    model: ModelWrapper, 
    tokenizer, 
    user_prompt, 
    max_new_tokens=100, 
    model_name='llama-2', 
    use_stop_texts=True, 
    temperature=0.01,
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
        with torch.no_grad():
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


        # 打印结果
        print(logits_dict, max_key)
        return max_key

def get_embedding(model: ModelWrapper, tokenizer, harmful_question, layer):
    mali_reps = get_reps(model=model, tokenizer=tokenizer, input_examples=[harmful_question], layers=layer)
    layer_reps = np.array(mali_reps[layer[0]])
    return layer_reps[0]

def load_classifier(path):
    with open(path, 'rb') as f:
        classifier = pickle.load(f)
    return classifier

def get_epsilon_dict(model: ModelWrapper, tokenizer, query, classifier_dict, cav_dict, p_0):
    epsilon_dict = {}
    t_cav_dict = {}
    
    for (layer_name, classifier) in classifier_dict.items():
        model.clear_cavs()
        # get cav
        if len(t_cav_dict) != 0:
            set_control_by_epsilon_dict(model, t_cav_dict, epsilon_dict)
        e = get_embedding(model, tokenizer, query, [layer_name])
        s_0 = np.log(p_0 / (1 - p_0))
        w = classifier.coef_[0]
        b = classifier.intercept_
        best_epsilon = ((s_0 - b - np.dot(w, e)) / np.linalg.norm(w))[0]
        if best_epsilon > 0:
            best_epsilon = 0
        epsilon_dict[layer_name] = best_epsilon / np.linalg.norm(w)
        t_cav_dict[layer_name] = cav_dict[layer_name]

    return epsilon_dict

def get_plus_epsilon_dict(model: ModelWrapper, tokenizer, query, classifier_dict, cav_dict, p_0 = 0.9):
    epsilon_dict = {}
    t_cav_dict = {}
    
    for (layer_name, classifier) in classifier_dict.items():
        model.clear_cavs()
        # get cav
        if len(t_cav_dict) != 0:
            set_control_by_epsilon_dict(model, t_cav_dict, epsilon_dict)
        e = get_embedding(model, tokenizer, query, [layer_name])
        s_0 = np.log(p_0 / (1 - p_0) + 1e-9)
        w = classifier.coef_[0]
        b = classifier.intercept_
        best_epsilon = ((s_0 - b - np.dot(w, e)) / np.linalg.norm(w))[0]
        if best_epsilon < 0:
            best_epsilon = 0
        epsilon_dict[layer_name] = best_epsilon / np.linalg.norm(w)
        t_cav_dict[layer_name] = cav_dict[layer_name]

    return epsilon_dict

def set_control_by_epsilon_dict(model: ModelWrapper, cav_dict, epsilon_dict):
    cavs = {}
    norm_sum = 0
    for (layer_name, epsilon) in epsilon_dict.items():
        cav = cav_dict[layer_name]
        cavs[layer_name] = epsilon * cav
        norm_sum += torch.norm(cavs[layer_name])
    
    model.clear_cavs()
    model.set_cavs(cavs)
    
    return norm_sum

def generate(
    base_model, 
    task_name, 
    cav_train_data_name,
    cav_p_0,
    start_layer, 
    end_layer, 
    cav_steer = 'minus',   # plus or minus
    max_new_tokens=32,
    with_pos_prefix=False, 
    prompt_prefix=None,
    num_test_queries=100,
    test_suffix='',
    use_gate=True
):

    ##### Load base model #####
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
            
    ##### Load CAVs and classifiers #####
    classifier_dict = {} 
    cav_dict = {}
    hook_layers = []
    
    for layer_id in range(start_layer, end_layer + 1, 1):
        layer_name = 'model.layers.{}'.format(str(layer_id))
        cav_dir = os.path.join('saved', cav_train_data_name, model_name, 'cavs')
        cav_file_path = os.path.join(cav_dir, 'layer_' + str(layer_id) + '_cav.pth')
        classifier_file_path = os.path.join(cav_dir, 'layer_' + str(layer_id) + '_model.pkl')                         
        cav = torch.load(cav_file_path)
        cav_dict[layer_name] = cav.to(model.device)
        classifier_dict[layer_name] = load_classifier(classifier_file_path)
        hook_layers.append(layer_name)

    model.register_forward_hooks(hook_layers)
    
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
        
    scores = check_human_values(prompts={cav_train_data_name: prompt_list_wo_tplt}, value=cav_train_data_name)
    
    if not use_gate:
        thresholds = VALUE_THRESHOLDS_default
        low_thresholds = VALUE_LOW_THRESHOLDS_default
    else:
        thresholds = VALUE_THRESHOLDS
        low_thresholds = VALUE_LOW_THRESHOLDS
    ## get prompts using scores
    prompt_list, prompt_list_origin, prompt_list_wo_tplt = load_test_data(
        task_name, 
        prefix=prefix, 
        model_name=model_name, 
        scores=scores[cav_train_data_name], 
        threshold=thresholds[cav_train_data_name],
        low_threshold=low_thresholds[cav_train_data_name],
        num_test_queries=num_test_queries,
        test_suffix=test_suffix
    )
    
    print('Filtered questions ratio:',sum((scores[cav_train_data_name] < thresholds[cav_train_data_name]) & (scores[cav_train_data_name] > low_thresholds[cav_train_data_name])) / scores[cav_train_data_name].shape[0])

    ##### Generate #####
    used_prompt_list = []
    processed_prompt_list = []
    generated = []
    control_label_list = []
    print(scores)
    for idx, (ques, oris_ques) in tqdm(enumerate(zip(prompt_list, prompt_list_origin)), total=len(prompt_list)):
        model.clear_cavs()
        if (scores is not None):
            print('score: {}, threshold: {}, low threshold: {}'.format(scores[cav_train_data_name][idx], thresholds[cav_train_data_name], low_thresholds[cav_train_data_name]))
        if (scores is not None) and ((scores[cav_train_data_name][idx] > thresholds[cav_train_data_name]) or scores[cav_train_data_name][idx] < low_thresholds[cav_train_data_name]):
            control_label_list.append(1)
            if cav_steer == 'plus':
                epsilon_dict = get_plus_epsilon_dict(model, tokenizer, ques, classifier_dict, cav_dict, p_0=cav_p_0)
            elif cav_steer == 'minus':
                epsilon_dict = get_epsilon_dict(model, tokenizer, ques, classifier_dict, cav_dict, p_0=cav_p_0)
            logging.info(epsilon_dict)
            
            set_control_by_epsilon_dict(model, cav_dict, epsilon_dict)
            gen_str = generate_(model, tokenizer, ques, max_new_tokens=max_new_tokens, model_name=model_name)
            used_prompt_list.append(ques)
        else:
            control_label_list.append(0)
            gen_str = generate_(model, tokenizer, oris_ques, max_new_tokens=max_new_tokens, model_name=model_name)
            used_prompt_list.append(oris_ques)
        generated.append(gen_str)
    torch.cuda.empty_cache()
    result_df = pd.DataFrame({'prompt':used_prompt_list, 'processed_prompt':prompt_list_wo_tplt, 'generated':generated, 'prompt_list_origin': prompt_list_origin, 'control_label': control_label_list})
    return result_df

def generate_mmlu(
    base_model, 
    cav_train_data_name,
    cav_p_0,
    start_layer, 
    end_layer, 
    mmlu: MMLU,
    cav_steer = 'minus',   # plus or minus
    max_new_tokens=32,
    with_pos_prefix=False, 
    prompt_prefix=None,
):

    ##### Load base model #####
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
            
    ##### Load CAVs and classifiers #####
    classifier_dict = {} 
    cav_dict = {}
    hook_layers = []
    
    for layer_id in range(start_layer, end_layer + 1, 1):
        layer_name = 'model.layers.{}'.format(str(layer_id))
        cav_dir = os.path.join('saved', cav_train_data_name, model_name, 'cavs')
        cav_file_path = os.path.join(cav_dir, 'layer_' + str(layer_id) + '_cav.pth')
        classifier_file_path = os.path.join(cav_dir, 'layer_' + str(layer_id) + '_model.pkl')                         
        cav = torch.load(cav_file_path)
        cav_dict[layer_name] = cav.to(model.device)
        classifier_dict[layer_name] = load_classifier(classifier_file_path)
        hook_layers.append(layer_name)

    model.register_forward_hooks(hook_layers)
    
    if with_pos_prefix and (prompt_prefix is not None):
        prefix = prompt_prefix
    else:
        prefix = None
        
        
    prompts, golden_labels = mmlu.get_pairs()
    
    
    if '_' in cav_train_data_name:
        cav_train_data_name = cav_train_data_name.split('_')[0]
        
    scores_mmlu = check_human_values_mmlu(prompts=prompts, value=cav_train_data_name)
    for k in prompts.keys():
        prompts[k] = add_template_and_prefix(
            prompts=prompts[k], 
            prefix=prefix, 
            model_name=model_name, 
            scores=scores_mmlu[k][cav_train_data_name], 
            threshold=VALUE_THRESHOLDS[cav_train_data_name],
            low_threshold=VALUE_LOW_THRESHOLDS[cav_train_data_name],
            use_tplt=False,
        )

    ##### Generate #####
    full_answers = {}
    for k in prompts.keys():
        full_answers[k] = []
        prompt_list = prompts[k]
        for idx, ques in tqdm(enumerate(prompt_list), total=len(prompt_list)):
            model.clear_cavs()
            if (scores_mmlu is not None):
                print('score: {}, threshold: {}, low threshold: {}'.format(scores_mmlu[k][cav_train_data_name][idx], VALUE_THRESHOLDS[cav_train_data_name], VALUE_LOW_THRESHOLDS[cav_train_data_name]))
            if (scores_mmlu is not None) and ((scores_mmlu[k][cav_train_data_name][idx] > VALUE_THRESHOLDS[cav_train_data_name]) or (scores_mmlu[k][cav_train_data_name][idx] < VALUE_LOW_THRESHOLDS[cav_train_data_name])):
                if cav_steer == 'plus':
                    epsilon_dict = get_plus_epsilon_dict(model, tokenizer, ques, classifier_dict, cav_dict, p_0=cav_p_0)
                elif cav_steer == 'minus':
                    epsilon_dict = get_epsilon_dict(model, tokenizer, ques, classifier_dict, cav_dict, p_0=cav_p_0)
                logging.info(epsilon_dict)
                
                set_control_by_epsilon_dict(model, cav_dict, epsilon_dict)
            gen_str = generate_(model, tokenizer, ques, max_new_tokens=max_new_tokens, model_name=model_name,
                                return_logits=True)
            full_answers[k].append(gen_str)

    torch.cuda.empty_cache()
    
    overall_accuracy = mmlu.calc_accuracies(prompts, full_answers, golden_labels, scores_mmlu, value=cav_train_data_name)
    
    for row in mmlu.predictions.iterrows():
        print(f"Question: {row[1]['Question']}\nAnswer: {row[1]['Answer']}\nGolden: {row[1]['Golden']}\nCorrect: {row[1]['Correct']}\nValue Score: {row[1]['Value Score']}\n")
    return overall_accuracy

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