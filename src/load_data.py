import pandas as pd
from src.utils import *

TOXICITY_POS_PREFIX = "The following text does not perpetuate negative stereotypes, is not threatening or sexually explicit, and does not contain profane language."
TOXICITY_NEG_PREFIX = "The following text perpetuates negative stereotypes, is threatening or sexually explicit, or contains profane language."

SENTIMENT_POS_PREFIX = "The following text is designed to be positive and friendly. It promotes a supportive and encouraging tone, fosters goodwill, and is free from any negative or harmful language."
SENTIMENT_NEG_PREFIX = "The following text is designed to express negative or hateful sentiments. It promotes a hostile or discouraging tone, fosters ill will, and contains harmful, derogatory, or offensive language."

TRADITION_POS_PREFIX = "The following text reflects traditional values, unconditional support and adherence to all old ways and rules."
TRADITION_NEG_PREFIX = "The following text reflects the abandonment and contempt for traditional values, and the unconditional opposition and resistance to all old ways and rules."



def get_chat_template(model_name):
    if 'llama-3-8b-Instruct' in model_name:
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>
 
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    elif 'llama-2-7b-chat' in model_name:
        return '[INST] {} [/INST]'
    elif 'qwen2.5' in model_name.lower():
        return "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

def load_cav_train_data(cav_data_name, model_name=None): 

    if ('tradition' in cav_data_name) or ('conformity' in cav_data_name) or ('power' in cav_data_name) or ('achievement' in cav_data_name) or ('hedonism' in cav_data_name) or ('stimulation' in cav_data_name) or ('security' in cav_data_name)  or ('benevolence' in cav_data_name) or ('universalism' in cav_data_name) or ('self-direction' in cav_data_name):
        df = pd.read_csv(f'data/value/{cav_data_name}.csv')
        pos_prompts = df['question_1'].tolist() # tradition
        neg_prompts = df['question_2'].tolist() # untradition
        
        pos_prompts_processed = []
        neg_prompts_processed = []
        chat_template = get_chat_template(model_name)
        for prompt in pos_prompts[:]:
            p = prompt.strip('"')
            pos_prompts_processed.append(p)
        for prompt in neg_prompts[:]:
            p = prompt.strip('"')
            neg_prompts_processed.append(p)
        
        pos_prompts = [s.strip('"') for s in pos_prompts_processed]
        neg_prompts = [s.strip('"') for s in neg_prompts_processed]
        
        return pos_prompts, neg_prompts
    
    else: 
        raise ValueError(f"CAV data {cav_data_name} not recognized")


def load_test_data(task_name, return_tag=False, prefix=None, model_name='', scores=None, threshold=None, low_threshold=None, num_test_queries=100, test_suffix=''): 
    if ('tradition' in task_name) or ('conformity' in task_name) or ('power' in task_name) or ('achievement' in task_name) or ('hedonism' in task_name) or ('stimulation' in task_name) or ('security' in task_name)  or ('benevolence' in task_name) or ('universalism' in task_name) or ('self-direction' in task_name):
        if '_' in task_name:
            task_name = task_name.split('_')[0]
        df = pd.read_csv(f'/home/xxx/GCAV/data/value/test_{task_name}{test_suffix}.csv')
        all_prompts = df['text'].tolist()
        prompt_list = []
        prompt_list_origin = []
        prompt_list_wo_tplt = []
        chat_template = get_chat_template(model_name)
        
        for idx, prompt in enumerate(all_prompts[:num_test_queries]):
            p = prompt.strip('"')
            prompt_list_wo_tplt.append(p)
            prompt_list_origin.append(chat_template.format(p))
            if (prefix is not None):
                if ((scores is not None) and ((scores[idx] > threshold) or (scores[idx] < low_threshold))) or (scores is None):
                    p = f"{prefix}\n{p}"
            prompt_list.append(chat_template.format(p))
        [print(p) for p in prompt_list[:5]]
        return prompt_list, prompt_list_origin, prompt_list_wo_tplt
    else: 
        raise ValueError(f"Task {task_name} not recognized")
    
    
def add_template_and_prefix(model_name, prompts, prefix=None, scores=None, threshold=None, low_threshold=None, use_tplt=True):
    if use_tplt:
        chat_template = get_chat_template(model_name)
    else:
        chat_template = '{}'
    prompt_list = []
    for idx, prompt in enumerate(prompts[:]):
        p = prompt.strip('"')
        if (prefix is not None):
            if ((scores is not None) and ((scores[idx] > threshold) or (scores[idx] < low_threshold))) or (scores is None):
                p = f"{prefix}\n{p}"
        prompt_list.append(chat_template.format(p))
        
    [print(p) for p in prompt_list[:5]]
    return prompt_list
