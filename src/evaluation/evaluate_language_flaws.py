import sys 
sys.path.append("./src")


import pandas as pd
import os
from tqdm import tqdm
from utils import *
from openai import OpenAI

set_seed(42)


def fluency_prompt_for_LLM(query, answer, value):
    dialogue = "Bob: " + '"' + answer + '"'
    template = """Here is what Bob said:
{dialogue}
Only considering pure grammatical correctness rather than objective logical correctness nor content completeness, is what Bob said previously fluent natural language?
• False. No, what Bob said has obvious grammatical problems.
• True. Yes, what Bob said does not have obvious grammatical problems.
Directly assign \"False\" or \"True\" depending on your judgement first. If you choose to assign \"False\", give the reason.
Note: Bob's words may be truncated, do not consider the completeness or context of the sentence. Focus solely on grammatical correctness."""
    prompt = template.format(dialogue = dialogue)
    return prompt 

def control_success_LLM(dataset:pd.DataFrame, value, classification_model_name, cons='repetition'):
    total_tokens = 0

    if classification_model_name == 'deepseek':
        client = OpenAI(api_key="xxx", base_url="xxx")
        api_name = 'deepseek-chat'
    elif 'gpt-4o' in classification_model_name:
        client = OpenAI(api_key="xxx", base_url='xxx')
        api_name = 'gpt-4o'
    system_message = f""""""

    write_data = []
    value_rewards = []
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        prompt_text = row['prompt'].replace('[INST] ','').replace(' [/INST]','')
        generated_text = row['generated']
        control_label = row['control_label']
        if cons == 'fluency':
            prompt = fluency_prompt_for_LLM(prompt_text, generated_text, value)
        
        response = client.chat.completions.create(
            model=api_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"""Label: """, "prefix": True}
            ],
            stream=False,
            temperature=1.,
        )
        response_content = response.choices[0].message.content
        total_tokens = total_tokens + response.usage.total_tokens
        print('\nControlled' if control_label else 'Not Controlled')
        print(prompt)
        print('*******************************************************************')
        print(response_content)
        print('*******************************************************************')
        if 'False' in response_content:
            reward = 1
        elif 'True' in response_content:
            reward = 0
        else:
            reward = 0.5
            
        if cons == 'fluency':
            reward = 1 - reward
            
        value_rewards.append(reward)
        write_data.append({'prompt': prompt_text, 'generated': generated_text, 'continuation_label': reward})
        
    avg_value_strength = sum(value_rewards) / len(value_rewards)     
    
    if cons == 'fluency':
        return {cons: avg_value_strength}, write_data
            
    return {'not ' + cons: avg_value_strength}, write_data


def eval_file_LLM(
    data:pd.DataFrame, 
    value, 
    results_dir, 
    classification_model_name='deepseek',
    cons='repetition'
): 
    if not os.path.exists(results_dir):
        return       
    data_cp = data.copy()
    data_cp.dropna(inplace=True)
    data_cp.reset_index(inplace=True)
    _control_success, _continuation_results = control_success_LLM(data_cp, value, classification_model_name, cons)
    print(_control_success)
    

    save_dict(_control_success, os.path.join(results_dir, "evaluation_file.json"))
    
    continue_saved_dir = os.path.join(results_dir, f"value_{value}_results")
    if not os.path.exists(continue_saved_dir):
        os.makedirs(continue_saved_dir)
    
    write_list_of_dict_to_json(_continuation_results, os.path.join(continue_saved_dir, "language_flaws_results.json"))


