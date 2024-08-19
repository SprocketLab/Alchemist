import os
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, CSVLoader, WikipediaLoader

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

def get_docs_for_rag(external_knowledge_base_path, link_paths):

    print("##############################\n")
    docs = []
    
    ### webpages ###
    for weblink in link_paths["weblink"]:
        print(f"Loading web content from {weblink}")
        loader = WebBaseLoader(weblink)
        docs += loader.load()

    ### webpdf ###
    for webpdf in link_paths["webpdf"]:
        print(f"Loading web pdf content from {webpdf}")
        loader = PyPDFLoader(webpdf)
        docs += loader.load()
        
    ### wikipages ###
    for wikilink in link_paths["wikipage"]:
        print(f"Loading wikipedia content from {wikilink}")
        loader = WikipediaLoader(weblink)
        docs += loader.load()
    
    ### local csv file ###
    for csvfile_path in link_paths["local_csv_file"]:
        local_csv = os.path.join(external_knowledge_base_path, csvfile_path)
        print(f"Loading local csv content from {local_csv}")
        loader = CSVLoader(local_csv)
        docs += loader.load()
        
    ### local txt file ###
    for textfile_path in link_paths["local_txt_file"]:
        local_txt = os.path.join(external_knowledge_base_path, textfile_path)
        print(f"Loading local text content from {local_txt}")
        loader = TextLoader(local_txt)
        docs += loader.load()
        
    ### local pdf file ###
    for pdffile_path in link_paths["local_pdf_file"]:
        local_pdf = os.path.join(external_knowledge_base_path, pdffile_path)
        print(f"Loading local pdf content from {local_pdf}")
        loader = PyPDFLoader(local_pdf)
        docs += loader.load()
            
    return docs
    
def custom_input(prompt):
    user_input = input(prompt)
    if user_input == "exit":
        exit("Quitting script")
    return user_input

def collect_args():

    print("Welcome to Alchemist system. We are helping you to produce labels automatically. Type exit at any time to quit.")
    args = {}

    args["current_path"] = os.getcwd()

    with open(os.path.join(args["current_path"], "config.json"), "r") as f:
        option_table = json.load(f)

    ###########################################################
    task_enter = custom_input("\nSpecify the dataset you are playing with. We have several default choices: \n\
        a. youtube (spam review classification), \n\
        b. sms (spam text classification), \n\
        c. imdb (sentiment classification), \n\
        d. yelp (sentiment classification), \n\
        e. agnews (topic classificaiton), \n\
        f. medabs (topic classification), \n\
        g. cancer (topic classification), \n\
        h. reddit (topic classification), \n\
        i. french (sentiment classification), \n\
        j. finance (sentiment classification), \n\
        (leave blank for default 'youtube'), \n\
        Enter: ").strip().lower() or "a"

    # i. trec (question classification), \n\
    # j. spouse (relation classificaiton), \n\
    # k. cdr (relation classification), \n\
    # l. semeval (relation classification), \n\
    # m. chemprot (relation classification), \n\
    
    if task_enter not in option_table["dataset"]:
        raise NotImplementedError("You are playing with a new dataset!")
        
    args["dataset"] = option_table["dataset"][task_enter]
    args['dataset_LF_saved_path'] = os.path.join(option_table["LF_saved_dir"], args["dataset"])

    args['exp_result_saved_path'] = os.path.join(args["current_path"], option_table["exp_result_saved_path"], args["dataset"])

    with open(os.path.join(args["current_path"], "prompt_template.json"), "r") as f:
        args["prompt_template"] = json.load(f)[args["dataset"]]
            
    ###########################################################
    mode_enter = custom_input("\nSelect the mode to generate labeling functions for your dataset. Choices: \n\
        a: ScriptoriumWS\n\
        b: Alchemist-without-RAG\n\
        c: Alchemist-with-RAG\n\
        (leave blank for default 'ScriptoriumWS')\n\
        Enter: ").strip().lower() or "a"
    
    if mode_enter not in ["a", "b", "c"]:
        raise ValueError("Select invalid mode!")

    args["mode"] = option_table["mode"][mode_enter]

    ###########################################################
    code_llm_enter = custom_input("\nSelect the model to use for code generation. Choices: \n\
        a: gpt-3.5-turbo (gpt-3.5-turbo-0125), \n\
        b: gpt-4 (gpt-4-0613), \n\
        c: claude 2.1 (claude-2.1), \n\
        d: claude 3 Sonnet (claude-3-sonnet-20240229), \n\
        (leave blank for default 'gpt-3.5-turbo')\n\
        Enter: ").strip().lower() or "a"

    if code_llm_enter not in ["a", "b", "c", "d"]:
        raise ValueError("Select invalid codellm!")
        
    args['codellm'] = option_table["codellm"][code_llm_enter]
        
    ###########################################################
    if mode_enter == "b":
        
        llm_enter = custom_input("\nSelect the model to use for prior knowledge generation. Choices: \n\
            a: gpt-3.5-turbo (gpt-3.5-turbo-0125), \n\
            b: gpt-4 (gpt-4-0613), \n\
            c: claude 2.1 (claude-2.1), \n\
            d: claude 3 Sonnet (claude-3-sonnet-20240229), \n\
            (leave blank for default 'gpt-3.5-turbo')\n\
            Enter: ").strip().lower() or "a"

        # e: llama2-7b, \n\
        
        args['llm'] = option_table["llm"][llm_enter]
        
        # if llm_enter == "e":
        #     args["llm_path"] = "/hdd1/llama_v2_hf/llama2-7b-chat-hf/"

    ###########################################################
    if mode_enter == "c":

        llm_enter = custom_input("\nSelect the model to use for prior knowledge generation. Choices: \n\
            a: gpt-3.5-turbo (gpt-3.5-turbo-0125), \n\
            b: gpt-4 (gpt-4-0613), \n\
            c: claude 2.1 (claude-2.1), \n\
            d: claude 3 Sonnet (claude-3-sonnet-20240229), \n\
            (leave blank for default 'gpt-3.5-turbo')\n\
            Enter: ").strip().lower() or "a"

        # e: llama2-7b, \n\
        
        args['llm'] = option_table["llm"][llm_enter]
        
        # if llm_enter == "e":
        #     args["llm_path"] = "/hdd1/llama_v2_hf/llama2-7b-chat-hf/"
        
        ###########################################################
        args['rag_embedding_model'] = option_table["rag_embedding_model"]
        args["rag_external_knowledge_base"] = os.path.join(args['current_path'], option_table['rag_external_knowledge_base'], args['dataset'])
        rag_json_file_path = os.path.join(args["rag_external_knowledge_base"], args['dataset'] + ".json")
        
        rag_link_input = custom_input(f"\nSpecify the path to file your external knowledge\n\
            (leave blank for default file path {rag_json_file_path})\n\
            Enter: ").strip() or None

        if rag_link_input == None:
            if os.path.isfile(rag_json_file_path):
                with open(rag_json_file_path, "r") as f:
                    external_dict = json.load(f)
                args['rag_links_path'] = external_dict
            else:
                raise FileNotFoundError("Cannot locate your rag file!")
        else:
            raise NotImplementedError("Please make your rag file!")

        ###########################################################
        docs = get_docs_for_rag(args["rag_external_knowledge_base"], args['rag_links_path'])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=args['rag_embedding_model'])
        vectorstore = FAISS.from_documents(texts, embeddings)
        args["vectorstore"] = vectorstore
    
    return args
