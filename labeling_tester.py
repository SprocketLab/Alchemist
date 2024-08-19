import os
import sys
import warnings
warnings.filterwarnings("ignore")

from labeling import Labeler

def main(dataset, mode, model, heuristic_mode=None):

    args = {
        'dataset': f'{dataset}', \
        'dataset_LF_saved_path': f'/hdd1/Alchemist_Data/{dataset}', \
        'exp_result_saved_path': f'/home/{os.getlogin()}/RAGCODE/exp_log/{dataset}', \
        'mode': f'{mode}', \
        'codellm': f'{model}', \
        'llm': f'{model}', \
        'prior_type': f'{heuristic_mode}',
        'LF_saving_exact_dir': f'/hdd1/Alchemist_Data/{dataset}/{mode}/{model}/{heuristic_mode or ""}'
    }
    print("********************")
    
    labeler = Labeler(args)
    labeler.run()

if __name__ == "__main__":
    
    # print("python3 labeling_tester.py [dataset name] [mode] [model] [heuristic type]")
    # print("Mode Options: 'ScriptoriumWS', 'Alchemist-without-RAG', 'Alchemist-with-RAG'")
    # print("Model Options: 'gpt-3.5-turbo', 'gpt-4', 'claude-2.1', 'claude-3-sonnet-20240229'")
    # print("Heuristic Types: 'labeling heuristics, rules, and guidance', 'keywords', 'dataset and class description', '5 data examples for each class'")
    print(sys.argv)
    if sys.argv[2] == "ScriptoriumWS":
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    