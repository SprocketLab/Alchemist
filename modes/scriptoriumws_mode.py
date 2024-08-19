import os
from openai import OpenAI
from anthropic import Anthropic
from config import custom_input
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from modes.base_mode import BaseMode

class ScriptoriumWSMode(BaseMode):
    
    def __init__(self, args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature):
        
        super().__init__(args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature)
        self.codellm_choice = self.args["codellm"]
        self.final_prompt = None
        self.synthesized_labeling_function = None
        self.estimated_cost = 0

    def gpt_inference_call(self):
        
        client = OpenAI(api_key = OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model= self.codellm_choice,
            temperature=1.0,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": self.codellm_system_prompt},
                {"role": "user", "content": self.final_prompt}
            ]
        )
        
        return completion

    def claude_inference_call(self):
        
        client = Anthropic(api_key = ANTHROPIC_API_KEY)
        completion = client.messages.create(
            model=self.codellm_choice,
            temperature=0.7,
            max_tokens=1000,
            system=self.codellm_system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": self.final_prompt}]}
            ]
        )
        
        return completion
        
    def run(self):
        
        self.final_prompt = self.mission_statement + "\n" + self.labeling_instruction + "\n" + self.function_signature
        
        print(f"\nFinal Prompt for CodeLLM ({self.codellm_choice}): \n\'\'\'\n{self.final_prompt}\n\'\'\'\n")

        print("##############################\n")
        if self.codellm_choice == "gpt-3.5-turbo" or self.codellm_choice == "gpt-4":
            completion = self.gpt_inference_call()
            self.calculate_cost(input_tok=completion.usage.prompt_tokens, output_tok=completion.usage.completion_tokens, model=self.codellm_choice)
            response = completion.choices[0].message.content
            
        elif self.codellm_choice == "claude-2.1" or self.codellm_choice == "claude-3-sonnet-20240229":
            completion = self.claude_inference_call()
            self.calculate_cost(input_tok=completion.usage.input_tokens, output_tok=completion.usage.output_tokens, model=self.codellm_choice)
            response = completion.content[0].text

        elif self.codellm_choice == "llama2-7b":
            raise NotImplementedError("To Be Added.")
            
        self.synthesized_labeling_function = self.extract_LF(response)
        
        if self.synthesized_labeling_function is None:
            return None
        
        print(f"\nYour synthesized labeling function is:\n{self.synthesized_labeling_function}\n")
        
        print("##############################\n")
        save_option = custom_input("Do you want to save this labeling function to file? (y/n):\n\
    (leave blank for default 'Yes')\n\
    Enter: ").strip().lower() or "y"
        
        if save_option == "y":
            self.save()
            
            ws_option = custom_input("\nDo you want to run label model and end model with synthesized LFs? (y/n):\n\
    (leave blank for default 'No')\n\
    Enter: ").strip().lower() or "n"
            
            if ws_option == "y":
                return "LABEL TIME"
                
        return None

