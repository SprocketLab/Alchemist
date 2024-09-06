# The Alchemist: Automated Labeling 500x CHEaper Than LLM Data Annotators

Currently, three modes are supported to generate labeling programs :<br />
1. **ScriptoriumWS mode**: One stage: LLM (your choice) is used to generate labeling functions for the given dataset.<br />
2. **Alchemist without RAG mode**: Two stages. In the first stage, it prompts a language model (such as GPT or Claude) to generate heuristics by utilizing prior knowledge along with the user's mission statement. In the second stage, it uses these generated heuristics to prompt a CodeLLM, creating labeling functions for the given dataset.<br /> 
3. **Alchemist with RAG mode**: Two stages. In the first stage, it pulls relevant information from a vector store, which holds pre-embedded documents or data (related to the dataset). This knowledge is used to generate heuristics. In the second stage, the model uses those heuristics to create labeling functions with the help of CodeLLM.<br />

### To Run the Code<br />

1. Create .env file in the RAGCODE folder and add `export OPENAI_API_KEY= #your API key#`  in the file<br />
2. Run the `python main.py` command on the terminal. This should start an interactive command line interface. <br />

### User Inputs

#### To include
1. **Dataset and the Task Description** <br />
Initially, you will describe your dataset and the labeling task at hand. This ensures that the generated labeling functions are aligned with your specific requirements.<br />

### Included [Code in `config.py`]<br />
Upon running the main, you will be prompted to provide various inputs through the terminal. Here’s what to expect:<br />

- **Mode Selection**: Choose the mode that best fits your needs:<br />
    - ScriptoriumWS mode
    - Alchemist without RAG mode
    - Alchemist with RAG mode<br />

### CODE Structure
1. `main.py` : to run the code - which starts up an interactive command line and launches different modes based on user inputs
2. `config.py` :  Manages the initial setup by capturing user inputs to configure the application accordingly. 
3. `executor.py` : instantiates the corresponding mode object and initiates the execution process tailored to that mode.
4. `modes` : contained code for different modes.
    - `base_mode.py` : parent class to all the modes
    - `scriptoriumws_mode.py`
    - `alchemist_without_RAG_mode.py`
    - `alchemist_with_RAG_mode.py`
   
