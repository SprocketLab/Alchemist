# The Alchemist: Automated Labeling 500x CHEaper Than LLM Data Annotators

Currently, three modes are supported to generate labeling programs :<br />
1. **Plain mode**: One stage. just prompt CodeLLM.<br />
2. **In-context learning without RAG mode**: Two Stages. Prompt LLM to generate heuristics first using prior knowledge. Use CodeLLM to generate labeling programs using these heuristics.<br /> 
3. **In-context learning with RAG mode**: Two stages. prompt prior knowledge with links, textbooks, and wiki first then merge into query prompt to ask CodeLLM.<br />

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
    - Plain mode
    - In-context learning without RAG mode
    - In-context learning with RAG mode<br />

**Note** - For Plain mode, only one model will be used

### CODE Structure
1. `main.py` : to run the code - which starts up an interactive command line and launches different modes based on user inputs
2. `config.py` :  Manages the initial setup by capturing user inputs to configure the application accordingly. 
3. `executor.py` : instantiates the corresponding mode object and initiates the execution process tailored to that mode.
4. `modes` : contained code for different modes.
    - `base_mode.py` : parent class to all the modes
    - `plain_mode.py`
    - `in-context_without_RAG_mode.py`
    - `in-context_with_RAG_mode.py`
   
