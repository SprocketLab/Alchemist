import os
import json
from config import custom_input

from modes.alchemist_with_RAG_mode import AlchemistWithRAGMode
from modes.alchemist_without_RAG_mode import AlchemistWithoutRAGMode
from modes.scriptoriumws_mode import ScriptoriumWSMode

class Executor:
    
    def __init__(self, args):
        self.args = args
        self.default_prompt = self.args["prompt_template"]

        self.codellm_system_prompt = self.get_codellm_system_prompt()
        self.mission_statement = self.get_mission_statement()
        self.labeling_instruction = self.get_labeling_instruction()
        self.function_signature = self.get_function_signature()
        
        self.labeler_system_prompt = None
        self.llm_system_prompt = None
        self.prior_knowledge_query = None

    def get_codellm_system_prompt(self):
        system_prompt = self.default_prompt['codellm_system_prompt']
        return system_prompt
        
    def get_mission_statement(self):
        mission = custom_input(f"\nWrite your mission statement prompt.\n\
    (leave blank for default '{self.default_prompt['mission_statement']}')\n\
    Enter: ").strip() or None
        if mission == None:
            mission = self.default_prompt['mission_statement']
        print(f"Your mission statement is: '{mission}'.")
        return mission

    def get_labeling_instruction(self):
        instruction = self.default_prompt['labeling_instruction']
        return instruction
   
    def get_function_signature(self):
        function_name = self.default_prompt['function_signature']
        return function_name

    def get_llm_system_prompt(self):
        system_prompt = self.default_prompt['llm_system_prompt']
        return system_prompt

    def get_prior_knowledge_query(self):

        prior_type_dict = {
            "a": "labeling heuristics, rules, and guidance",
            "b": "keywords",
            "c": "dataset and class description",
            "d": "5 data examples for each class"
        }
            
        type_enter = custom_input("\nSelect the type of prior knowledge to generate. Choices: \n\
    a: labeling heuristics, rules, and guidance, \n\
    b: keywords, \n\
    c: dataset and class description, \n\
    d: 5 data examples for each class, \n\
    (leave blank for default 'labeling heuristics, rules, and guidance')\n\
    Enter: ").strip().lower() or "a"

        if type_enter not in prior_type_dict:
            raise ValueError("Select invalid type!")
            
        prior_type = prior_type_dict[type_enter]
        self.args["prior_type"] = prior_type
        prior_knowledge_prompt = self.default_prompt['prior_knowledge'].replace("[prior knowledge]", prior_type)
        
        prior_query = custom_input(f"\nWrite your prior knowledge query.\n\
    (leave blank for default '{prior_knowledge_prompt}')\n\
    Enter: ").strip() or None

        if prior_query == None:
            prior_query = prior_knowledge_prompt
        print(f"Your query for prior knowledge is: '{prior_query}'.\n")
        
        return prior_query

    def get_labeler_system_prompt(self):
        labeler_system_prompt = self.default_prompt['labeler_system_prompt']
        return labeler_system_prompt
    
    def execute_mode(self):

        if self.args["mode"] == "ScriptoriumWS":
            response = self.run_scriptorium_ws_mode()
            
        elif self.args["mode"] == "Alchemist-without-RAG":
            response = self.run_alchemist_without_rag_mode()
            
        elif self.args["mode"] == "Alchemist-with-RAG":
            response = self.run_alchemist_with_rag_mode()

        if response == "LABEL TIME":
            return 1
        return 0

    def run_scriptorium_ws_mode(self):

        print("\n##############################")
        print("\nRunning ScriptoriumWS Mode")
        
        mode_obj = ScriptoriumWSMode(
            args = self.args,
            codellm_system_prompt = self.codellm_system_prompt,
            mission_statement = self.mission_statement,
            labeling_instruction = self.labeling_instruction,
            function_signature = self.function_signature,
        )
        
        return mode_obj.run()

    def run_alchemist_without_rag_mode(self):

        print("\n##############################")
        print("\nRunning Alchemist without RAG mode")
        
        self.llm_system_prompt = self.get_llm_system_prompt()
        self.prior_knowledge_query = self.get_prior_knowledge_query()
        
        mode_obj = AlchemistWithoutRAGMode(
            args = self.args,
            codellm_system_prompt = self.codellm_system_prompt,
            mission_statement = self.mission_statement,
            labeling_instruction = self.labeling_instruction,
            function_signature = self.function_signature,
            llm_system_prompt = self.llm_system_prompt,
            prior_knowledge_query = self.prior_knowledge_query
        )
        
        return mode_obj.run()

    def run_alchemist_with_rag_mode(self):

        print("\n##############################")
        print("\nRunning Alchemist with RAG mode")
        
        self.llm_system_prompt = self.get_llm_system_prompt()
        self.prior_knowledge_query = self.get_prior_knowledge_query()
        
        mode_obj = AlchemistWithRAGMode(
            args = self.args,
            codellm_system_prompt = self.codellm_system_prompt,
            mission_statement = self.mission_statement,
            labeling_instruction = self.labeling_instruction,
            function_signature = self.function_signature,
            llm_system_prompt = self.llm_system_prompt,
            prior_knowledge_query = self.prior_knowledge_query
        )

        return mode_obj.run()
