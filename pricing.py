import re
import os
import sys

def get_LF_file_paths(LF_directory):
        
    file_path_collection = []
    
    for f in os.listdir(LF_directory):
        file_path = os.path.join(LF_directory, f)
        if os.path.isfile(file_path) and f.endswith(".py"):
            file_path_collection.append(file_path)
            
    return file_path_collection


def get_total_cost(dataset, mode, model, heuristic_mode=None):
    
    total_cost = 0
    
    if heuristic_mode is None:
        LF_directory = os.path.join(os.path.sep, "hdd1", "Alchemist_Data", dataset, mode, model)
    else:
        LF_directory = os.path.join(os.path.sep, "hdd1", "Alchemist_Data", dataset, mode, model, heuristic_mode)

    file_paths = get_LF_file_paths(LF_directory)
    
    for file_path in file_paths:
        with open(file_path, "r") as f:
            code_string = f.read()
        pattern = re.compile(r'\$\[(.*?)\]')
        matches = pattern.findall(code_string)
        total_cost += float(matches[0])
    
    print("=====================================")
    print(f'Total cost of {len(file_paths)} LFs: ${total_cost}')
    print("=====================================")


if __name__ == "__main__":
    # usage: python3 pricing.py [dataset name] [mode] [model] [heuristic mode]
    
    if sys.argv[2] == "ScriptoriumWS":
        get_total_cost(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        get_total_cost(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])