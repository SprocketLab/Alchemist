# The ALCHEmist: Automated Labeling 500x CHEaper Than LLM Data Annotators

[![arXiv](https://img.shields.io/badge/-Paper-blue?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2407.11004) [![Project Site](https://img.shields.io/badge/🌐-Project_Site-green?labelColor=gray)](https://zihengh1.github.io/alchemist/)

Large pretrained models can be used as annotators, helping replace or augment crowdworkers and enabling distilling generalist models into smaller specialist models. Unfortunately, this comes at a cost: **employing top-of-the-line models often requires paying thousands of dollars for API calls, while the resulting datasets are static and challenging to audit.** 

To address these challenges, we propose a simple alternative: **rather than directly querying labels from pretrained models, we task models to generate programs that can produce labels.**

Our system, **Alchemist**, obtains comparable to or better performance than large language model-based annotation in a range of tasks for a fraction of the cost: on average, improvements amount to a 12.9% enhancement while the total labeling costs across all datasets are reduced by a factor of approximately 500x.

Currently, three modes are supported to generate labeling programs :<br />
1. **ScriptoriumWS mode**: One stage: LLM (your choice) is used to generate labeling programs for the given dataset.<br />
2. **Alchemist without RAG mode**: Two stages. In the first stage, it prompts a language model (such as GPT or Claude) to generate **heuristics by utilizing prior knowledge along with the user's mission statement.** In the second stage, it uses these generated heuristics to prompt a CodeLLM, creating labeling programs for the given dataset.<br /> 
3. **Alchemist with RAG mode**: Two stages. It combine RAG system with your given information to generate heuristics. Then the model uses those heuristics to create labeling programs with the help of CodeLLM.<br />


### To Run the Code<br />
1. Install `wrench` from [Wrench Benchmark](https://github.com/JieyuZ2/wrench?tab=readme-ov-file). 
2. Create an `.env` file in the Alchemist folder and add `export OPENAI_API_KEY= <your API key>` to it.
    - If desired, modify the `LF_saved_dir` field in `config.json` to the directory that you wish the generated labeling functions to be saved to.
3. Run the `python main.py` command on the terminal. This should start an interactive command line interface. See the following section on user inputs.<br />
4. Type `exit` at any time to stop execution of the program.

### Datasets and Generated Programs
We share the youtube dataset, finance, and french datasets as examples and our generated programs used in the paper [here](https://drive.google.com/drive/folders/12cJUdDcbc3NKDTsHc0SiWtM2BTvk-v4T?usp=sharing).


### User Inputs

#### To include
1. **Dataset and the Task Description**
Initially, you will describe your dataset and the labeling task at hand. This ensures that the generated labeling functions are aligned with your specific requirements.<br />

### Included [Code in `config.py`]<br/>
Upon running `main.py`, you will be prompted to provide various inputs through the terminal. Here’s what to expect:<br/>

- **Dataset Selection**: Choose following example dataset that you'd like to generate labeling functions for:<br />
    - youtube (spam review classification)  
    - sms (spam text classification)  
    - imdb (sentiment classification)  
    - yelp (sentiment classification)  
    - medabs (topic classification)  
    - cancer (topic classification)  
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
4. `modes`: Directory that contains code for different modes. See above for descriptions of the last three modes.
    - `base_mode.py`: Parent class to all the modes.
    - `scriptoriumws_mode.py`
    - `alchemist_without_RAG_mode.py`
    - `alchemist_with_RAG_mode.py`
5. `pricing.py`: Used to parse the generated labeling function files and returns the total cost by dataset, mode, mode, and heuristic mode.

## Citation
Please cite our paper if you find the repository helpful.
```
@inproceedings{huang2024the,
    title={The {ALCHE}mist: Automated Labeling 500x {CHE}aper than {LLM} Data Annotators},
    author={Tzu-Heng Huang and Catherine Cao and Vaishnavi Bhargava and Frederic Sala},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=T0glCBw28a}
}
```