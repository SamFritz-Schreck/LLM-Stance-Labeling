'''
packages to install: transformers accelerate bitsandbytes datasets peft trl
'''
import pandas as pd, os, logging, time
import accelerate, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  
from peft import LoraConfig
from datasets import Dataset

'''
Specify any helper functions
'''
def formatting_prompts_func(example):
    # formatter for the data so it works with SFT
    output_texts = []
    if model_name.split("/")[1] == "Mistral-7B-Instruct-v0.1":
        for i in range(len(example[context_col])):
            if example['dataset'][i] == 'semeval':
                text = f"{example[context_col][i]}\n entity: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'election':
                text = f"{example[context_col][i]}\n politcian: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'covid':
                text = f"{example[context_col][i]}\n belief: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'pheme':
                text = f"{example[context_col][i]}\n rumor: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'wt':
                text = f"{example[context_col][i]}\n merger: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'srq':
                text = f"{example[context_col][i]}\n social media post: {example['target_text'][i]}\n topic: {example['event'][i]}\n statement: {example['response_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            output_texts.append(text)
    else:
        for i in range(len(example[context_col])):
            if example['dataset'][i] == 'semeval':
                text = f"{example[context_col][i]}\n entity: {example['event'][i]}\n statement: {example['full_text'][i]}\n stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'election':
                text = f"{example[context_col][i]}\n politcian: {example['event'][i]}\n statement: {example['full_text'][i]}\n stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'covid':
                text = f"{example[context_col][i]}\n belief: {example['event'][i]}\n statement: {example['full_text'][i]}\n stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'pheme':
                text = f"{example[context_col][i]}\n rumor: {example['event'][i]}\n statement: {example['full_text'][i]}\n stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'wt':
                text = f"{example[context_col][i]}\n merger: {example['event'][i]}\n statement: {example['full_text'][i]}\n stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'srq':
                text = f"{example[context_col][i]}\n social media post: {example['target_text'][i]}\n topic: {example['event'][i]}\n statement: {example['response_text'][i]}\n stance: {example['train_stance'][i]}"
            output_texts.append(text)
    return output_texts

'''
Set up the training run
'''
# Data set up
training_data_filename = "/home/jovyan/Army Youtube Comments/all_data_training.pkl"
context_col = 'context_analyze' # different prompt schemes. Options are: context_question and context_analyze
test_datasets = ["wtwt"] # full set of benchmark datasets: ["semeval", "election", "pheme", "covid", "srq", "wtwt"]

# model set up
# Read in personal access token  
with open('/home/jovyan/huggingface_token.txt', 'r') as file:  
    token = file.read().strip()

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
sample = False
quantization = False

# Set up logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO)

'''
Read in the dataset
'''
# Read the dataframe
df = pd.read_pickle(training_data_filename)  

# Do a sample (for debugging purposes)
if sample:
    df = df.sample(n=100)

logging.info("Number of data points in dataset: {}".format(len(df)))

'''
Run the training. For each dataset split, convert the dataset into a Dataset object,
set up the training arguments, the directories, the model and the trainer
'''
for i in range(len(test_datasets)):
    start_time = time.time()
    train_df = df[df['dataset'] != test_datasets[i]]

    logging.info("---------------fold {}---------------".format(i))
    logging.info("test dataset for this fold: {}".format(test_datasets[i]))
    logging.info("train datasets for this fold: {}".format(train_df['dataset'].unique()))

    # Convert the data into a Huggingface Dataset object
    dataset = Dataset.from_pandas(train_df)

    output_dir = "/home/jovyan/LLM-Stance-Labeling/"+test_datasets[i]+"_checkpoints"
    logging_dir = "/home/jovyan/LLM-Stance-Labeling/"+test_datasets[i]+"_training_logs"
    final_model_save_dir = "/home/jovyan/LLM-Stance-Labeling/"+test_datasets[i]+"_fine_tuned_model"

    '''
    Create the model, tokenizer, and peft config
    '''
    # Read in the tokenizer and set the quantization configuration
    if quantization:
        quantization_config = BitsAndBytesConfig(
                load_in_16bit=True
            )
    else:
        quantization_config= None

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # Read in the model and use device map='auto' to spread the device across multiple GPUs
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
        quantization_config=quantization_config, token=token, device_map="auto")

    if not tokenizer.pad_token:  
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    tokenizer.padding_side = 'right'

    # Set up peft for efficient tuning
    peft_config = LoraConfig(  
        r=256,  
        lora_alpha=16,  
        lora_dropout=0.1,  
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    '''
    Set up the trainer and train
    '''
    # Define the training arguments  
    training_args = TrainingArguments(  
        output_dir=output_dir,
        num_train_epochs=2,
        auto_find_batch_size=True,
        #per_device_train_batch_size=32,
        #per_device_eval_batch_size=64,
        warmup_steps=100,
        learning_rate= 1e-5,
        gradient_accumulation_steps=12,
        weight_decay=0.01,
        save_steps = 100,
        logging_steps = 10,
        logging_dir=logging_dir,
        fp16=True, # mixed-precision training
    )

    # set up data collator for SFT trainer
    response_template = "stance:"  
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

    # create the SFT trainer
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset, 
        formatting_func=formatting_prompts_func,  
        data_collator=collator,  
        peft_config=peft_config,
        max_seq_length = 800
    )  

    # trainer.train(resume_from_checkpoint = True)
    trainer.train()

    trainer.save_model(final_model_save_dir)

    elapsed_time = time.time() - start_time  
    logging.info("Time elapsed for this training run: {} seconds".format(elapsed_time))
