import argparse
import pickle
from src.utils import *

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modelwrapper import ModelWrapper
import pandas as pd 
from tqdm import tqdm

from src.load_data import load_cav_train_data

from src.cav import get_reps
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_cavs(positive_reps, negative_reps, train_size=1.0, delta=None, random_state=42):
    positive_labels = np.ones ((len(positive_reps),))  
    negative_labels = np.zeros((len(negative_reps),))  

    X = np.vstack((positive_reps, negative_reps))
    Y = np.concatenate((positive_labels, negative_labels))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=random_state, train_size=train_size)
    if delta is None:
        log_reg = LogisticRegression(solver='saga', max_iter=10000)
        log_reg.fit(x_train, y_train)
    else:
        log_reg = LogisticRegression(penalty='l1', C=delta, solver='saga', max_iter=10000)
        log_reg.fit(x_train, y_train)

    predictions_test = log_reg.predict(x_test)
    predictions_train = log_reg.predict(x_train)
    acc_test = accuracy_score(y_test, predictions_test)
    acc_train = accuracy_score(y_train, predictions_train)
    cav = log_reg.coef_[0]
    
    return cav, log_reg, acc_train, acc_test


def run(args):
    model_name = args.base_model.split('/')[-1]
    cav_saved_dir = os.path.join('saved', args.cav_data_name, model_name,  'cavs')

    if not os.path.exists(cav_saved_dir):
        os.makedirs(cav_saved_dir)
        
    ##### Load train data #####
    
    pos_prompts, neg_prompts = load_cav_train_data(args.cav_data_name, model_name)
    
    pos_prompts = pos_prompts[:args.num_train_pairs]
    neg_prompts = neg_prompts[:args.num_train_pairs]
    
    [print(s) for s in pos_prompts[:5]]
    [print(s) for s in neg_prompts[:5]]
    print(len(pos_prompts))
    
    

    hf_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map={'': 'cuda:0'}).cuda().eval()
    model = ModelWrapper(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
    ) 
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token

    if '7b' in model_name.lower():
        if 'llama' in model_name.lower():
            end_layer = 32
        elif 'qwen' in model_name.lower():
            end_layer = 28
    elif '13b' in model_name.lower():
        if 'llama' in model_name.lower():
            end_layer = 40
    elif '14b' in model_name.lower():
        if 'qwen' in model_name.lower():
            end_layer = 48
    elif '8b' in model_name.lower():
        if 'llama' in model_name.lower():
            end_layer = 32
    
    # print()
    layer_list = []
    for layer_id in range(0, end_layer):
        layer_name = 'model.layers.{}'.format(str(layer_id))
        layer_list.append(layer_name)
    
    print(layer_list)
            
    model.register_forward_hooks(layer_list)
    
    print(cav_saved_dir)
    positive_reps = get_reps(model=model, tokenizer=tokenizer, input_examples=pos_prompts, layers=layer_list)
    negative_reps = get_reps(model=model, tokenizer=tokenizer, input_examples=neg_prompts, layers=layer_list)
    
    print('reps got!')
    
    acc_list = []
    random_state = 0

    for layer_id in tqdm(range(0, end_layer, 1)):

        layer = 'model.layers.{}'.format(str(layer_id))
        pos_reps, neg_reps = positive_reps[layer], negative_reps[layer]

        cav, log_reg, acc_train, acc_test = get_cavs(pos_reps, neg_reps, train_size=0.7, delta=None, random_state=random_state)

        cav = torch.from_numpy(cav)
        cav_filename = os.path.join(cav_saved_dir, 'layer_' + str(layer_id) + '_cav.pth')
        torch.save(cav, cav_filename)
        model_filename = os.path.join(cav_saved_dir, 'layer_' + str(layer_id) + '_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(log_reg, f)
        acc_list.append(acc_test)
        print('Layer: {}, Acc_train: {:.2f}, Acc_test: {:.2f}'.format(layer_id, acc_train, acc_test))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='Llama-2-7b-chat-hf', type=str)
    parser.add_argument('--cav_data_name', default='real_toxic', type=str)
    parser.add_argument('--num_train_pairs', default=100, type=int)
    args = parser.parse_args()

    run(args)