import os 
import json
import numpy as np 
import torch
import random

PERSPECTIVE_API_KEY = ""   
    
STOP_TEXTS = ["\n"]

POS_PREFIX = "The following text does not perpetuate negative stereotypes, is not threatening or sexually explicit, and does not contain profane language."
NEG_PREFIX = "The following text perpetuates negative stereotypes, is threatening or sexually explicit, or contains profane language."

LABEL_COLUMNS = ['Self-direction: thought','Self-direction: action','Stimulation','Hedonism','Achievement','Power: dominance','Power: resources','Face','Security: personal', 'Security: societal','Tradition','Conformity: rules','Conformity: interpersonal','Humility','Benevolence: caring','Benevolence: dependability','Universalism: concern','Universalism: nature','Universalism: tolerance','Universalism: objectivity']

VALUE_THRESHOLDS = {
    "achievement": 0.6,
    "stimulation": 0.02,
    "hedonism": 0.007,
    "self-direction": 0.5,
    "power": 0.06,
    "security": 0.2,
    "tradition": 0.08,
    "conformity": 0.013,
    "benevolence": 0.2,
    "universalism": 0.39,
}
VALUE_LOW_THRESHOLDS = {
    "achievement": 0.,
    "stimulation": 0.0,
    "hedonism": 0.00,
    "self-direction": 0.,
    "power": 0.0,
    "security": 0.,
    "tradition": 0.0,
    "conformity": 0.0,
    "benevolence": 0.,
    "universalism": 0.3,
}


def create_saved_dir_name(args):
    model_name = args.base_model.split('/')[-1]
    if args.method == 'base_model':
        results_saved_path = os.path.join( args.results_dir,  args.method, args.task_name, model_name, "pos_prefix_"+ str(args.with_pos_prefix) )
    elif args.method == 'sft':
        results_saved_path = os.path.join( args.results_dir,  args.method, args.task_name, model_name, "pos_prefix_"+ str(args.with_pos_prefix) )
    elif args.method == 'caa': 
        results_saved_path = os.path.join(args.results_dir, args.method, args.task_name, model_name, 'coef_' + str(args.act_coef), 'start_'+str(args.start_layer)+'_end_'+str(args.end_layer) )  
    elif args.method == 'caa_multi':
        results_saved_path = os.path.join( args.results_dir,  args.method, args.task_name, args.concept_list, model_name, f"{str(args.act_coef_list)}", f"start_{args.start_layer}_end_{args.end_layer}" )
    elif args.method == 'cav': 
        results_saved_path = os.path.join(args.results_dir, args.method, args.task_name, model_name, f"cav_data_{args.cav_data_name}", 'cav_p_' + str(args.cav_p_0), 'start_'+str(args.start_layer)+'_end_'+str(args.end_layer) ) 
    elif args.method == 'cav_multi':
        results_saved_path = os.path.join(args.results_dir, args.test_data, args.method,  args.concept_list, model_name, f"cav_data_{args.cav_data_name}", 'cav_p_' + str(args.cav_p_0_list), 'start_'+str(args.start_layer)+'_end_'+str(args.end_layer) )
    else:
        raise ValueError(f"Method {args.method} not recognized")
    return results_saved_path


def save_dict(dict_, path_):
    if os.path.exists(path_):
        with open(path_, "r") as f:
            _results_ = json.load(f)
    else:
        _results_ = {}
    _results_.update(dict_)
    json.dump(_results_, open(path_, "w"), indent=4)
    
    
def write_list_of_dict_to_json(list_of_dict, write_file_path):
    with open(write_file_path, 'w') as f:
        for item in list_of_dict:
            json.dump(item, f)
            f.write('\n')


def log(function, message):
    function(message)
    


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_args(args, saved_dir):
    print(f"saving to {saved_dir}")
    file_path = os.path.join(saved_dir, 'running_args.json')
    args_dict = vars(args)
    with open(file_path, 'w') as f:
        json.dump(args_dict, f)