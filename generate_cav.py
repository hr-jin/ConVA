import argparse
import os
from src.utils import *
import logging
import warnings

from src.cav_gen import generate as generate_cav, generate_mmlu as generate_cav_mmlu
from src.baselines.base_model import generate as generate_base_model, generate_mmlu as generate_base_model_mmlu
from src.baselines.sft import generate as generate_sft

from src.baselines.caa import generate as generate_caa
from src.evaluation.evaluate_value import eval_file_LLM as eval_value_by_LLM
from src.evaluation.evaluate_language_flaws import eval_file_LLM as eval_language_flaws_by_LLM
import warnings


from mmlu import MMLU

MMLU_TASKS_LOWER = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']


seed = 42
set_seed(seed)

def run(args): 
    print(args)
    warnings.filterwarnings("ignore", category=UserWarning)


    model_name = args.base_model.split('/')[-1]
    results_saved_path = create_saved_dir_name(args)
    
    if os.path.exists(os.path.join(results_saved_path, 'result.csv')):
        print("Results already exist in ", results_saved_path)
            
    if not os.path.exists(results_saved_path):
        os.makedirs(results_saved_path)
    save_args(args, results_saved_path)
    # Set up logging to a file
    logging.basicConfig(filename=os.path.join(results_saved_path, 'logging.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    

    if args.method == 'cav':
        result_df = generate_cav(
            base_model=args.base_model, 
            task_name=args.task_name, 
            cav_train_data_name=args.cav_data_name,
            cav_p_0=args.cav_p_0,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            cav_steer=args.cav_steer,
            max_new_tokens=100,
            with_pos_prefix=args.with_pos_prefix, 
            prompt_prefix=args.prompt_prefix,
            num_test_queries=args.num_test_queries,
            test_suffix=args.test_suffix,
            use_gate=args.use_gate,
        )
    elif args.method == 'base_model':
        result_df = generate_base_model(
            args.base_model, 
            args.task_name, 
            args.with_pos_prefix, 
            args.prompt_prefix, 
            return_tag=args.return_tag,
            max_new_tokens=100,
            num_test_queries=args.num_test_queries,
            test_suffix=args.test_suffix
        )
    elif args.method == 'sft':
        result_df = generate_sft(
            args.base_model, 
            args.task_name, 
            args.lora_path,
            args.with_pos_prefix, 
            args.prompt_prefix, 
            return_tag=args.return_tag,
            max_new_tokens=100,
            num_test_queries=args.num_test_queries,
            test_suffix=args.test_suffix
        )
        
    elif args.method == 'caa':
        result_df = generate_caa(
            args.base_model, 
            args.task_name, 
            act_coef=args.act_coef,
            cav_train_data_name=args.cav_data_name,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            max_new_tokens=100,
            with_pos_prefix=args.with_pos_prefix, 
            prompt_prefix=args.prompt_prefix,
            num_test_queries=args.num_test_queries,
            test_suffix=args.test_suffix
        )
        
    
    else: 
        raise ValueError(f"Method {args.method} not recognized")
    
    print(f'saving results dataframe to {results_saved_path}')
    result_df.to_csv(os.path.join(results_saved_path, 'result.csv'), index=False, escapechar='\\', lineterminator="\n")
    
    
    if args.method in ['cav', 'caa', 'base_model', 'sft']: 
        for value in ['tradition', 'conformity', 'security', 'benevolence', 'universalism', 'power', 'self-direction', 'stimulation', 'hedonism', 'achievement']:
            if value in args.task_name: 
                
                classification_model_name = 'gpt-4o'
                
                torch.cuda.empty_cache()
                eval_value_by_LLM(result_df, args.task_name, results_saved_path, classification_model_name)
                
                torch.cuda.empty_cache()
                eval_language_flaws_by_LLM(result_df, args.task_name, results_saved_path, classification_model_name, 'fluency')
                
                torch.cuda.empty_cache()
                        
    torch.cuda.empty_cache()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='Llama-2-7b-chat-hf', type=str)
    parser.add_argument('--task_name', default='toxicity_toxic', type=str)
    parser.add_argument('--method', default='cav', type=str)
    parser.add_argument('--cav_data_name', default='real_toxic', type=str)
    parser.add_argument('--start_layer', default=16, type=int)
    parser.add_argument('--end_layer', default=31, type=int)
    parser.add_argument('--act_coef', default=2.0, type=float)
    parser.add_argument('--cav_p_0', default=1e-7, type=float)
    parser.add_argument('--results_dir', default='results/', type=str)
    parser.add_argument('--eval_based_model_path', default='', type=str)
    parser.add_argument('--cav_steer', default='minus', type=str, choices=['plus', 'minus'])
    parser.add_argument('--concept_list', default=None, type=str)
    parser.add_argument('--cav_steer_list', default=None, type=str)
    parser.add_argument('--test_data', default=None, type=str)
    parser.add_argument('--cav_p_0_list', default=None, type=float, nargs='+')
    parser.add_argument('--with_pos_prefix', action='store_true')
    parser.add_argument('--prompt_prefix', default=None, type=str)
    parser.add_argument('--return_tag', default=False, type=bool)
    parser.add_argument('--act_coef_list', default=None, type=float, nargs='+')
    parser.add_argument('--num_test_queries', default=100, type=int)
    parser.add_argument('--test_suffix', default="", type=str)
    parser.add_argument('--lora_path', default=None, type=str)
    parser.add_argument('--use_gate', default=1, type=int)
    



    args = parser.parse_args()
    
    
    run(args)