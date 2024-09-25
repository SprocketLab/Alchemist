# The Alchemist: Automated Labeling 500x CHEaper Than LLM Data Annotators

Large pretrained models can be used as annotators, helping replace or augment crowdworkers and enabling distilling generalist models into smaller specialist models. Unfortunately, this comes at a cost: employing top-of-the-line models often requires paying thousands of dollars for API calls, while the resulting datasets are static and challenging to audit. To address these challenges, we propose a simple alternative: rather than directly querying labels from pretrained models, we task models to generate programs that can produce labels. These programs can be stored and applied locally, re-used and extended, and cost orders of magnitude less. Our system, Alchemist, obtains comparable to or better performance than large language model-based annotation in a range of tasks for a fraction of the cost: on average, improvements amount to a 12.9% enhancement while the total labeling costs across all datasets are reduced by a factor of approximately 500x.

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

- **Dataset Selection**: Choose the dataset that you'd like to generate labeling functions for:<br />
    - youtube (spam review classification)  
    - sms (spam text classification)  
    - imdb (sentiment classification)  
    - yelp (sentiment classification)  
    - agnews (topic classification)  
    - medabs (topic classification)  
    - cancer (topic classification)  
    - reddit (topic classification)  
    - french (sentiment classification)  
    - finance (sentiment classification)

- **Mode Selection**: Choose the mode that best fits your needs:<br />
    - ScriptoriumWS mode
    - Alchemist without RAG mode
    - Alchemist with RAG mode<br />

- **LLM Selection for Code Generation**: Choose the model you'd like to use for labeling function code generation:<br />
    - gpt-3.5-turbo (gpt-3.5-turbo-0125)
    - gpt-4 (gpt-4-0613)
    - claude 2.1 (claude-2.1)
    - claude 3 Sonnet (claude-3-sonnet-20240229)<br />

- **LLM Selection for Prior Knowledge Generation**: If you selected "Alchemist without RAG mode" or "Alchemist with RAG mode", you will be prompted to select the model you'd like to use for prior knowledge generation:<br />
    - gpt-3.5-turbo (gpt-3.5-turbo-0125)
    - gpt-4 (gpt-4-0613)
    - claude 2.1 (claude-2.1)
    - claude 3 Sonnet (claude-3-sonnet-20240229)<br />

- **Specify RAG Path**: If you selected "Alchemist with RAG mode", you will be prompted to specify the file path to your external knowledge.<br />

- **Specify Heuristics for Generating Prior Knowledge**: If you selected "Alchemist without RAG mode" or "Alchemist with RAG mode", you must select which type of prior knowledge to generate:<br />
    - labeling heuristics, rules, and guidance
    - keywords
    - dataset and class description
    - 5 data examples for each class<br />


### Code Structure
1. `main.py`: Main file used to run the code that starts up an interactive command line and launches different modes based on user inputs.
2. `config.py`:  Manages the initial setup by capturing user inputs to configure the application accordingly. 
3. `executor.py`: Instantiates the corresponding mode object and initiates the execution process tailored to that mode.
4. `modes`: Directory that contains code for different modes.
    - `base_mode.py`: Parent class to all the modes
    - `scriptoriumws_mode.py`
    - `alchemist_without_RAG_mode.py`
    - `alchemist_with_RAG_mode.py`
5. `pricing.py`: Used to parse the generated labeling function files and returns the total cost by dataset, mode, mode, and heuristic mode.
