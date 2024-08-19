import re
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from openai import OpenAI
from anthropic import Anthropic
from config import custom_input
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from modes.base_mode import BaseMode

from langchain import HuggingFacePipeline

class AlchemistWithoutRAGMode(BaseMode):
    
    def __init__(self, args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature, llm_system_prompt, prior_knowledge_query):
        
        super().__init__(args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature)
        
        self.codellm_choice = self.args["codellm"]
        self.llm_choice = self.args["llm"]
        self.llm_system_prompt = llm_system_prompt
        self.prior_knowledge_query = prior_knowledge_query
        
        self.generated_prior_knowledge = None
        self.final_prompt = None
        self.synthesized_labeling_function = None

    def remove_empty_lines(self, result):
        return re.sub(r'\n\s*\n', '\n', result).strip()

    """
    def llama2_inference_call(self):
        
        llama2_tokenizer = AutoTokenizer.from_pretrained(self.args["llm_path"], local_files_only=True)
        llama2_model = AutoModelForCausalLM.from_pretrained(self.args["llm_path"], local_files_only=True).to("cuda:0")
        
        llama2_pipe = transformers.pipeline("text-generation", model=llama2_model, tokenizer=llama2_tokenizer, max_new_tokens=1024, torch_dtype=torch.float16, device=0)
        llama2_hf_pipe = HuggingFacePipeline(pipeline=llama2_pipe, model_kwargs={"temperature": 0.2})
        completion = llm_pipeline.invoke(self.llm_system_prompt + "\n" + self.prior_knowledge_query)
        
        return completion
    """
    
    def gpt_inference_call(self, model_choice, system_prompt, input_prompt):
        
        client = OpenAI(api_key = OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model= model_choice,
            temperature=0.7,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ]
        )
        
        return completion

    def claude_inference_call(self, model_choice, system_prompt, input_prompt):
        
        client = Anthropic(api_key = ANTHROPIC_API_KEY)
        completion = client.messages.create(
            model=model_choice,
            temperature=0.7,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": input_prompt}]}
            ]
        )
        
        return completion
    
    def run(self):

        if self.llm_choice == "gpt-3.5-turbo" or self.llm_choice == "gpt-4":
            completion = self.gpt_inference_call(self.llm_choice, self.llm_system_prompt, self.prior_knowledge_query)
            self.calculate_cost(input_tok=completion.usage.prompt_tokens, output_tok=completion.usage.completion_tokens, model=self.llm_choice)
            response = completion.choices[0].message.content
            
        elif self.llm_choice == "claude-2.1" or self.llm_choice == "claude-3-sonnet-20240229":
            completion = self.claude_inference_call(self.llm_choice, self.llm_system_prompt, self.prior_knowledge_query)
            self.calculate_cost(input_tok=completion.usage.input_tokens, output_tok=completion.usage.output_tokens, model=self.llm_choice)
            response = completion.content[0].text

        # elif self.codellm_choice == "llama2-7b":
        #     raise NotImplementedError("To Be Added.")
            
        self.generated_prior_knowledge = self.remove_empty_lines(response)
        
        self.final_prompt = "Here is useful information to follow for generating labeling function:\n" + self.generated_prior_knowledge + "\n" + self.mission_statement + "\n" + self.labeling_instruction + "\n" + self.function_signature
        
        print(f"\nFinal Prompt for CodeLLM ({self.codellm_choice}): \n\'\'\'\n{self.final_prompt}\n\'\'\'\n")

        print("##############################\n")
        if self.codellm_choice == "gpt-3.5-turbo" or self.codellm_choice == "gpt-4":
            completion = self.gpt_inference_call(self.codellm_choice, self.codellm_system_prompt, self.final_prompt)
            self.calculate_cost(input_tok=completion.usage.prompt_tokens, output_tok=completion.usage.completion_tokens, model=self.codellm_choice)
            response = completion.choices[0].message.content
            
        elif self.codellm_choice == "claude-2.1" or self.codellm_choice == "claude-3-sonnet-20240229":
            completion = self.claude_inference_call(self.codellm_choice, self.codellm_system_prompt, self.final_prompt)
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
