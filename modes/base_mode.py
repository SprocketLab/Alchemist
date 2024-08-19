import re
import os
import json

class BaseMode:
    
    def __init__(self, args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature):

        self.args = args
        self.codellm_system_prompt = codellm_system_prompt
        self.mission_statement = mission_statement
        self.labeling_instruction = labeling_instruction
        self.function_signature = function_signature
        self.input_tok = 0
        self.output_tok = 0

    def calculate_cost(self, input_tok, output_tok, model="gpt-3.5-turbo"):
        
        self.input_tok = self.input_tok + input_tok
        self.output_tok = self.output_tok + output_tok
        
        pricing_table = {
            "gpt-3.5-turbo": {
                "prompt": 0.0005,
                "completion": 0.0015,
            },
            "gpt-4": {
                "prompt": 0.03,
                "completion": 0.06,
            },
            "claude-2.1": {
                "prompt": 0.008,
                "completion": 0.024,
            },
            "claude-3-sonnet-20240229": {
                "prompt": 0.003,
                "completion": 0.015,
            },
        }
    
        try:
            model_pricing = pricing_table[model]
        except KeyError:
            raise ValueError("Invalid model specified")
    
        prompt_cost = self.input_tok * model_pricing['prompt'] / 1000
        completion_cost = self.output_tok * model_pricing['completion'] / 1000
    
        self.estimated_cost = round(prompt_cost + completion_cost, 6)

        self.token_report = f"# Accumulated tokens used: {self.input_tok} (prompt) + {self.output_tok} (completion) = [{self.input_tok + self.output_tok}] tokens in total."
        self.estimated_cost_report = f"# Total cost for this inference call: $[{self.estimated_cost:.4f}]."

        print(self.token_report)
        print(self.estimated_cost_report)

    def extract_LF(self, result):
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, result, re.DOTALL)
        if len(matches) == 0:
            return result
        return matches[0]
    
    def save(self):

        LF_saving_parent_dir = os.path.join(self.args['dataset_LF_saved_path'], self.args['mode'])
        if not os.path.exists(LF_saving_parent_dir):
            os.mkdir(LF_saving_parent_dir)

        LF_saving_parent_dir = os.path.join(LF_saving_parent_dir, self.args["codellm"])
        if not os.path.exists(LF_saving_parent_dir):
            os.mkdir(LF_saving_parent_dir)
            
        if "prior_type" not in self.args:
            LF_saving_exact_dir = LF_saving_parent_dir
        else:
            prior_type = self.args["prior_type"]
            LF_saving_exact_dir = os.path.join(LF_saving_parent_dir, prior_type)
            if not os.path.exists(LF_saving_exact_dir):
                os.mkdir(LF_saving_exact_dir)

        LF_count = 0
        for filename in os.listdir(LF_saving_exact_dir):
            if filename.endswith('.py'):
                LF_count += 1
                
        self.args["LF_saving_exact_dir"] = LF_saving_exact_dir
        LF_file_path = os.path.join(LF_saving_exact_dir, f"LF{LF_count + 1}.py")
        
        with open(LF_file_path, "w") as file:
            print(f"Synthesized LF is saving to {LF_saving_exact_dir} and named as 'LF{LF_count + 1}.py'.")
            file.write("\'\'\'\n" + self.final_prompt + "\n\'\'\'\n" + self.token_report + "\n" + self.estimated_cost_report + "\n\n" + self.synthesized_labeling_function + "\n")

    def run(self):
        raise NotImplementedError("The run method must be implemented by subclasses.")




















        