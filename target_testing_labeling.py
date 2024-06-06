'''
packages to install for WIRE environment:
pip install transformers langchain accelerate bitsandbytes einops
'''

import os, re, pandas as pd, numpy as np, ast, json, string, logging, sys, warnings, time
from pprint import pprint
from tqdm import tqdm

'''
Specify Dataset, Model, and Number of Runs. 
Decoder only models include: mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, 
microsoft/phi-2, 
meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Llama-2-70b-chat-hf, 
tiiuae/falcon-7b-instruct, tiiuae/falcon-40b-instruct
facebook/opt-2.7b, facebook/opt-6.7b
Encoder-decoder models include: declare-lab/flan-alpaca-gpt4-xl, declare-lab/flan-alpaca-xxl, google/flan-ul2
'''
#MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL = sys.argv[1]
with open('/home/jovyan/huggingface_token.txt', 'r') as file:  
    token = file.read().strip()

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset")

'''
Import DL packages
'''
import torch
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import SequentialChain, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import accelerate

from sklearn.metrics import classification_report

# Create a logger and set level to INFO.  
logger = logging.getLogger()  
logger.setLevel(logging.INFO)  
# Create a file handler for output file.  
handler = logging.FileHandler('labeling_logs/target_testing.log') 
# Add the file handler to logger  
logger.addHandler(handler) 

'''
Helper Functions for Post-processesing, Model creation, and Prompt Generation
'''

def post_process_results(results):
    """
    This function post-processes the results from a large language model to label text.

    Args:
        results (list): A list of strings, each string is a word from language model output.

    Returns:
        list: A list of strings, each string is a classification label ('disagree', 'neutral', 'agree').
    """

    # A list to store the prediction labels
    y_pred = []  

    # Words or phrases that indicate each stance category
    disagree_indicators = ['against', 'denies', 'critical', 'deny', 'neg', 'oppose', 'opposes']
    agree_indicators = ['support','supports', 'for', 'pro ', 'positive', 'agree', 'agrees']
    neutral_indicators = ['neutral', 'unrelated']

    # Iterate over LLM return in the results
    for word in results:  
        # Normalize the word to lower case and remove leading/trailing white spaces
        normalized_word = str(word).strip().lower()

        if any(indicator in normalized_word for indicator in disagree_indicators):
            # If the word is also found in agree_indicators or neutral_indicators, label it as 'neutral'
            if any(indicator in normalized_word for indicator in agree_indicators) or any(indicator in normalized_word for indicator in neutral_indicators):
                y_pred.append('neutral')
            else:
                y_pred.append('disagree')
        elif any(indicator in normalized_word for indicator in neutral_indicators):
            # If the word is also found in disagree_indicators or agree_indicators, label it as 'neutral'
            if any(indicator in normalized_word for indicator in disagree_indicators) or any(indicator in normalized_word for indicator in agree_indicators):
                y_pred.append('neutral')
            else:
                y_pred.append('neutral')
        elif any(indicator in normalized_word for indicator in agree_indicators):
            # If the word is also found in disagree_indicators or neutral_indicators, label it as 'neutral'
            if any(indicator in normalized_word for indicator in disagree_indicators) or any(indicator in normalized_word for indicator in neutral_indicators):
                y_pred.append('neutral')
            else:
                y_pred.append('agree')
        else:  
            # If no specific stance label is found, label it as 'neutral'
            y_pred.append('neutral')


    return y_pred

def create_standard_chain(llm, dataset):
    # context prompt
    #Classify the stance of the following reply as to whether it supports, denies, or neutral toward the statement.
    #Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. What is the stance of the following social media satement toward the following entity? Give the stance as for, against, neutral, or unrelated. Only return the stance and no other text.

    if dataset == 'semeval':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided person. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        person: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'election':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following social media statement and determine its stance towards the provided person. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        person: {event}
        statement: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    elif dataset == 'basil':
        context_template = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a text toward a certain, specified target. Analyze the following news story and determine its stance towards the provided person. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        politician: {event}
        news story: {statement}
        stance:'''

        context_prompt = PromptTemplate(
            input_variables=["event","statement"],
            template=context_template
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    return LLMChain(prompt=context_prompt, llm=llm)

def create_zero_shot_CoT_chain(llm, dataset):
    # zero-shot Chain-of-Thought Prompt
    if dataset == "semeval":
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (for, against, neutral, or unrelated) of the social media statement towards the person.
        person: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        person: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")
    
    elif dataset == 'election':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (for, against, neutral, or unrelated) of the social media statement towards the person.
        person: {event}
        statement: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        person: {event}
        statement: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

    elif dataset == 'basil':
        cot_template_1 = '''Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain the stance (for, against, neutral, or unrelated) of the news story towards the person.
        person: {event}
        news story: {statement}
        explanation:'''

        cot_prompt_1 = PromptTemplate(
            input_variables=["event","statement"],
            template=cot_template_1
        )

        cot_chain_1 = LLMChain(llm=llm, prompt=cot_prompt_1, output_key="stance_reason")

        cot_template_2 ='''Therefore, based on your explanation, {stance_reason}, what is the final stance? Respond with a single word: "supports", "denies", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
        person: {event}
        news story: {statement}
        stance:'''

        cot_prompt_2 = PromptTemplate(
            input_variables=["event","statement","stance_reason"],
            template=cot_template_2
        )

        cot_chain_2 = LLMChain(llm=llm, prompt=cot_prompt_2, output_key="label")

    else:
        raise ValueError(f"Unknown dataset '{dataset}' was given.")

    llm_chain = SequentialChain(
        chains=[cot_chain_1, cot_chain_2],
        input_variables = ["event", "statement"],
        output_variables=["label"]
    )

    return llm_chain


def create_llm(model, token):
    encoder_decoder_models =['google/flan-ul2', 'declare-lab/flan-alpaca'] 

    if any(model.startswith(base) for base in encoder_decoder_models):
        tokenizer = AutoTokenizer.from_pretrained(model, token=token)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16, #quantizing model
            token = token, #hugginface token for gated models
            device_map="auto",
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            framework = "pt",
            truncation = True,
            max_new_tokens=50
        )

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, token=token)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text = False, #only return the newly generated text
            torch_dtype=torch.float16, #quantizing model
            token = token, #hugginface token for gated models
            device_map="auto",
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            framework = "pt",
            truncation = True,
            max_new_tokens=50
        )

        llm = HuggingFacePipeline(pipeline=pipe)

    return llm

if __name__ == "__main__":
    ### read in dataset 
    # file_path = "/home/jovyan/LLM-Stance-Labeling/bias results/target_testing_dataset.csv"
    ## Read in previous dataset, in case a run gets interrupted
    file_path = "/home/jovyan/LLM-Stance-Labeling/bias results/target_testing_results_"+MODEL.split("/")[1]+".csv"
    df = pd.read_csv(file_path)

    ### Create the data transformation for testing
    data_transformation = {
        'normal': df.copy(),
        'underspecify': df.copy().replace({'event': {'Donald Trump': 'Trump', 'Hillary Clinton': 'Clinton'}}),
        'candidate_and_name': df.copy().replace({'event': {'Donald Trump': 'Presidential Candidate Donald Trump', 
                                                    'Hillary Clinton': 'Presidential Candidate Hillary Clinton'}}),
        'party': df.copy().replace({'event': {'Donald Trump': 'The Republican Presidential Candidate', 
                                        'Hillary Clinton': 'The Democratic Presidential Candidate'}}),
        'party_and_name': df.copy().replace({'event': {'Donald Trump': 'The Republican Presidential Candidate Donald Trump', 
                                                'Hillary Clinton': 'The Democratic Presidential Candidate Hillary Clinton'}}),
        'misspelling': df.copy().replace({'event': {'Donald Trump': 'Donold Trump', 'Hillary Clinton': 'Hillery Clinton'}})
    }

    ### create model
    llm = create_llm(MODEL, token)

    ### run test examples and log output
    # What is the stance of the following social media post toward the following entity? Give the stance as either for, against, or neutral. Only return the stance and no other text.
    # What is the stance of the following social media post toward the following entity? Classify the stance as for, against, neutral, or unrelated if the post is not about the entity. Only return the stance, and no other text.
    question = '''Analyze the following social media post and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    entity: U.S. Army
    post: @vondeveen If the Army wants to actually recruit people, maybe stop breaking people and actually prosecute sexual assualt #nomorewar.
    stance:'''
    logger.info(question+llm(question))

    question = '''Analyze the following social media post and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    entity: U.S. Army
    post: @artfulask I have never seen a pink-eared duck before. #Army
    stance:'''
    logger.info(question+llm(question))

    question = '''Analyze the following social media post and determine its stance towards the provided entity. Respond with a single word: "for", "against", "neutral", or "unrelated". Only return the stance as a single word, and no other text.
    entity: U.S. Army
    post: I think the @Army helped me become disciplined. I would have surely flunked out of college chasing tail if I didn't get some discipline there. #SFL
    stance:'''
    logger.info(question+llm(question))

    ### for num_runs, for each prompting scheme
    ### run the labeling and log the output counts and a classification report

    # create the chains
    standard_chain_semeval = create_standard_chain(llm, "semeval")
    standard_chain_election = create_standard_chain(llm, "election")
    standard_chain_basil = create_standard_chain(llm, "basil")
    zero_shot_cot_chain_semeval = create_zero_shot_CoT_chain(llm, "semeval")
    zero_shot_cot_chain_election = create_zero_shot_CoT_chain(llm, "election")
    zero_shot_cot_chain_basil = create_zero_shot_CoT_chain(llm, "basil")

    # run the labelings
    start_time = time.time()
    for transformation_type, temp_df in data_transformation.items():
        
        ## standard context prompt
        logger.info("\n\n running standard conext prompt")
        results = []
        for index, row in tqdm(temp_df.iterrows()):
            if row['dataset'] == 'semeval':
                results.append(standard_chain_semeval.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))
            elif row['dataset'] == 'election':
                results.append(standard_chain_election.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))
            elif row['dataset'] == 'basil':
                results.append(standard_chain_basil.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['context_preds_raw_'+transformation_type] = results
        df['context_preds_'+transformation_type] = y_pred
        # save an interim result, in case the run dies
        df.to_csv("/home/jovyan/LLM-Stance-Labeling/bias results/target_testing_results_"+MODEL.split("/")[1]+"_1.csv", index=False)
        
        ## zero-shot CoT prompt
        logger.info("\n\n running zero-shot CoT prompt")
        results = []
        for index, row in tqdm(temp_df.iterrows()):
            if row['dataset'] == 'semeval':
                results.append(zero_shot_cot_chain_semeval.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))
            elif row['dataset'] == 'election':
                results.append(zero_shot_cot_chain_election.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))
            elif row['dataset'] == 'basil':
                results.append(zero_shot_cot_chain_basil.run(event=row['event'], statement=str(row['full_text']).replace('\n', ' ') ))

        # post-process and check results
        logger.info("raw results: {}\n".format(np.unique(results, return_counts=True)))
        y_pred = post_process_results(results)
        logger.info("processed results: {}\n".format(np.unique(y_pred, return_counts=True)))
        report = classification_report(df['stance'], y_pred, zero_division=0)
        logger.info(report)

        # store results in dataframe
        df['zero_shot_cot_preds_raw_'+transformation_type] = results
        df['zero_shot_cot_preds_'+transformation_type] = y_pred
        # save an interim result, in case the run dies
        df.to_csv("/home/jovyan/LLM-Stance-Labeling/bias results/target_testing_results_"+MODEL.split("/")[1]+"_1.csv", index=False)
    

    elapsed_time = time.time() - start_time  
    logger.info("Time elapsed for this run: {} seconds".format(elapsed_time))

    # Save out the final results
    df.to_csv("/home/jovyan/LLM-Stance-Labeling/bias results/target_testing_results_"+MODEL.split("/")[1]+"_1.csv", index=False)
