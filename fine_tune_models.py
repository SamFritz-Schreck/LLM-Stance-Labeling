'''
packages to install: transformers accelerate bitsandbytes datasets peft trl
'''
import pandas as pd, os, logging, time
from sklearn.metrics import f1_score
import accelerate, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from transformers import EarlyStoppingCallback, IntervalStrategy
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  
from peft import LoraConfig, PrefixTuningConfig
from datasets import Dataset
from huggingface_hub import login

'''
Specify any helper functions
'''
def formatting_prompts_func(example):
    # formatter for the data so it works with SFT
    output_texts = []
    if MODEL_NAME.split("/")[1] == "Mistral-7B-Instruct-v0.1":
        for i in range(len(example[context_col])):
            if example['dataset'][i] == 'semeval':
                text = f"{example[context_col][i]}\n entity: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'election':
                text = f"{example[context_col][i]}\n politcian: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'covid':
                text = f"{example[context_col][i]}\n belief: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'pheme':
                text = f"{example[context_col][i]}\n rumor: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'wtwt':
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
            elif example['dataset'][i] == 'wtwt':
                text = f"{example[context_col][i]}\n merger: {example['event'][i]}\n statement: {example['full_text'][i]}\n stance: {example['train_stance'][i]}"
            elif example['dataset'][i] == 'srq':
                text = f"{example[context_col][i]}\n social media post: {example['target_text'][i]}\n topic: {example['event'][i]}\n statement: {example['response_text'][i]}\n stance: {example['train_stance'][i]}"
            output_texts.append(text)
    return output_texts
'''
Set up the training run
'''
# Data set up
training_data_filename = "/home/jovyan/LLM-Stance-Labeling/all_data_training.pkl"
context_col = 'context_analyze' # different prompt schemes. Options are: context_question and context_analyze
test_datasets = ["wtwt"] # full set of benchmark datasets: ["semeval", "election", "pheme", "covid", "srq", "wtwt"]

# model set up
# Read in personal access token and login. Needed for Gated Models
with open('/home/jovyan/huggingface_token.txt', 'r') as file:
    token = file.read().strip()
login(token)


# model options: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-Instruct-v0.1, declare-lab/flan-alpaca-xxl, google/flan-ul2
MODEL_NAME = "google/flan-ul2"
PEFT_TYPE = 'adapter' #options are prefix and adapter
SAMPLE = False

# Set up logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO)

'''
Read in the dataset
'''
# Read the dataframe
df = pd.read_pickle(training_data_filename)  

# Do a sample (for debugging purposes)
if SAMPLE:
    df = df.sample(n=100)

logging.info("Number of data points in dataset: {}".format(len(df)))

'''
Run the training. For each dataset split, convert the dataset into a Dataset object,
set up the training arguments, the directories, the model and the trainer
'''
for i in range(len(test_datasets)):
    start_time = time.time()
    train_df = df[df['dataset'] != test_datasets[i]].sample(frac=1)
    val_df = df[df['dataset'] == test_datasets[i]]

    logging.info("---------------fold {}---------------".format(i))
    logging.info("test dataset for this fold: {}".format(test_datasets[i]))
    logging.info("train datasets for this fold: {}".format(train_df['dataset'].unique()))

    # Convert the data into a Huggingface Dataset object
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    output_dir = "/home/jovyan/LLM-Stance-Labeling/fine_tuned_"+PEFT_TYPE+"_models/"+test_datasets[i]+"_"+MODEL_NAME.split("/")[1]+"_checkpoints"
    logging_dir = "/home/jovyan/LLM-Stance-Labeling/fine_tuned_"+PEFT_TYPE+"_models/"+test_datasets[i]+"_"+MODEL_NAME.split("/")[1]+"_training_logs"
    final_model_save_dir = "/home/jovyan/LLM-Stance-Labeling/fine_tuned_"+PEFT_TYPE+"_models/"+test_datasets[i]+"_"+MODEL_NAME.split("/")[1]+"_fine_tuned_model"

    '''
    Create the model, tokenizer, and peft config
    '''
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not tokenizer.pad_token:  
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    tokenizer.padding_side = 'right'

    # Read in the model and use device map='auto' to spread the device across multiple GPUs
    if MODEL_NAME.split("/")[1] in ['flan-alpaca-xxl', 'flan-alpaca-xl', 'flan-ul2', 'flan-alpaca-gpt4-xl', "flan-t5-base", "flan-t5-large", "flan-t5-xxl"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")
        task_type = "SEQ_2_SEQ_LM"
        target_modules = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
        mixed_precision = False
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")
        task_type = "CASUAL_LM"
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        mixed_precision = True

    # Set up peft for efficient tuning
    if PEFT_TYPE == 'adapter':
        peft_config = LoraConfig(  
            r=32,  
            lora_alpha=1,  
            lora_dropout=0.1,  
            bias="none",
            task_type=task_type, #note: this parameter is needed for proper training behavior
            target_modules = target_modules
        )

    elif PEFT_TYPE == 'prefix':
        peft_config = PrefixTuningConfig(
            task_type=task_type,
            base_model_name_or_path=MODEL_NAME,
            num_virtual_tokens=30,
            encoder_hidden_size=1024,
            prefix_projection=True,
            inference_mode=False
        )

    '''
    Set up the trainer and train
    '''
    # Define the training arguments  
    training_args = TrainingArguments(  
        output_dir=output_dir,
        num_train_epochs=2,
        auto_find_batch_size=True,
        # per_device_train_batch_size=32,
        # per_device_eval_batch_size=64,
        warmup_steps=100,
        learning_rate= 1e-4,
        gradient_accumulation_steps=6,
        weight_decay=0.01,
        save_steps = 200,
        logging_steps = 10,
        logging_dir=logging_dir,
        fp16=mixed_precision, # mixed-precision training
        load_best_model_at_end=True,
        eval_steps = 100,
        evaluation_strategy = IntervalStrategy.STEPS
    )

    # set up data collator for SFT trainer
    response_template = "stance:"  
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

    # create the SFT trainer
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompts_func,  
        data_collator=collator,  
        peft_config=peft_config,
        max_seq_length = 800,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train(resume_from_checkpoint = True)
    # trainer.train()

    trainer.save_model(final_model_save_dir)

    elapsed_time = time.time() - start_time  
    logging.info("Time elapsed for this training run: {} seconds".format(elapsed_time))
