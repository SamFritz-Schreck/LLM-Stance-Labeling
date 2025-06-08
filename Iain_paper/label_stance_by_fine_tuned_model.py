'''
packages to install for WIRE: transformers accelerate peft langchain
'''
# Import necessary libraries  
import pandas as pd  
import numpy as np  
import string, sys, time, warnings, os
import multiprocessing as mp
from tqdm import tqdm
tqdm.pandas()
#Import DL libraries
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from accelerate import PartialState  
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain  
from peft import PeftConfig, PeftModel
  
# Define environment variables  
DATASET = sys.argv[2] # full set of benchmark datasets: ["semeval", "election", "pheme", "covid", "srq", "wtwt"]
MODEL = sys.argv[3] # base models: ["flan-alpaca-xxl", "flan-ul2", "Llama-2-7b-hf", "Mistral-7B-Instruct-v0.1"]
MODEL_PATH = "/home/jovyan/LLM-Stance-Labeling/fine_tuned_adapter_models/"+DATASET+"_"+MODEL+"_fine_tuned_model"
GPU_DEVICES = [int(x) for x  in sys.argv[1].split(',')]

warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset")

def create_context_analyze_chain(dataset, llm):
    # context prompt
    #Classify the stance of the following reply as to whether it supports, denies, or neutral toward the statement.
    #Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media satement toward the following entity? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.

    if dataset == 'semeval':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        entity: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'election':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided politcian. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        politcian: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'srq':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following reply and determine its stance toward the provided social media statement about the provided event. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        event: {event}
        statement: {statement}
        reply: {reply}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement", "reply"],
            template=context_template
        )
    elif dataset == 'pheme':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided rumor. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        rumor: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset =='covid':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided belief about COVID. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        belief: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset =='wtwt':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided merger actually happening. Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        merger: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    return LLMChain(prompt=context_prompt, llm=llm)
  
# Define a function that applies the pipeline to a dataframe    
def apply_pipeline(args):
    start_time = time.time()  
    # Unpack the arguments  
    (data, gpu_index) = args  
  
    # Load configuration, model, and tokenizer  
    config = PeftConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # Load the base pretrained model
    if config.base_model_name_or_path in ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-Instruct-v0.1"]:
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map=gpu_index)
        # Add the adapter
        model = PeftModel.from_pretrained(model, MODEL_PATH).merge_and_unload(progressbar=True)  
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,   
                        eos_token_id=tokenizer.eos_token_id,  
                        pad_token_id=tokenizer.eos_token_id,  
                        max_new_tokens=50)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map=gpu_index)
        # Add the adapter
        model = PeftModel.from_pretrained(model, MODEL_PATH).merge_and_unload(progressbar=True)  
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,   
                        eos_token_id=tokenizer.eos_token_id,  
                        pad_token_id=tokenizer.eos_token_id,  
                        max_new_tokens=50)

    # create the langchain from the LLM and appropriate prompting scheme
    llm = HuggingFacePipeline(pipeline=pipe)
    llm_chain = create_context_analyze_chain(DATASET, llm)

    # Apply the pipeline to the data 
    if DATASET == 'srq':
        data["finetuned_pred_raw"] = data.progress_apply(lambda row: llm_chain.run(event=row['event'], statement=str(row['target_text']).replace('\n', ' ') , reply=str(row['response_text']).replace('\n', ' ')), axis=1)  
    else: 
        data["finetuned_pred_raw"] = data.progress_apply(lambda row: llm_chain.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ')), axis=1)  
    
    elapsed_time = time.time() - start_time  
    print("labeling sub-process completed for GPU: {}, Elapsed time in seconds: {}".format(gpu_index, elapsed_time))
    return data  



if __name__ == '__main__':
    # Set the start method for multiprocessing  
    mp.set_start_method('spawn')

    # Load data    
    df = pd.read_pickle("/home/jovyan/LLM-Stance-Labeling/all_data_training.pkl")
    df = df[df['dataset'] == DATASET]
    print(DATASET)
    print("length of dataframe: {}".format(len(df)))
    
    # Split the data into chunks for parallel processing  
    dfs = np.array_split(df, len(GPU_DEVICES))  
    
    # Use a pool of processes to apply the function in parallel    
    with mp.Pool(processes=len(GPU_DEVICES)) as pool:    
        df_list = pool.map(apply_pipeline, zip(dfs, GPU_DEVICES))  

    # Combine the results into a single DataFrame  
    df = pd.concat(df_list)  
    
    # Ensure original order    
    df.sort_index(inplace=True)  
    
    # Save the DataFrame to a CSV file  
    df.to_csv(os.path.join("fine_tuned_results", DATASET+"_"+MODEL+"_fine_tuned_results.csv"), index=False)