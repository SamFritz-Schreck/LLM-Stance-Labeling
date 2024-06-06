import pandas as pd, os
import accelerate, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  
from peft import LoraConfig  
from datasets import Dataset  
  
# Read in personal access token  
with open('/home/jovyan/LLM-Stance-Labeling/personal_hugginface_token.txt', 'r') as file:  
    token = file.read().strip()
    
# Read the dataframe
df = pd.read_pickle("/home/jovyan/LLM-Stance-Labeling/all_data_training.pkl")  

# Convert the DataFrame to a Hugging Face Dataset  
dataset = Dataset.from_pandas(df)

# Read in the model and tokenizer use device map='auto' to spread the device across multiple GPUs
'''
quantization_config = BitsAndBytesConfig(
        load_in_16bit=True
    )
'''
quantization_config= None

model_name = "mistralai/Mistral-7B-Instruct-v0.1"  
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
    quantization_config=quantization_config, token=token, device_map="auto")

if not tokenizer.pad_token:  
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

tokenizer.padding_side = 'right'

# Define the training arguments  
training_args = TrainingArguments(  
    output_dir='/home/jovyan/LLM-Stance-Labeling/adapter_model_tuning_results',
    num_train_epochs=2,
    auto_find_batch_size=True,
    #per_device_train_batch_size=32,
    #per_device_eval_batch_size=64,
    warmup_steps=100,
    learning_rate= 1e-5,
    gradient_accumulation_steps=12,
    weight_decay=0.01,
    save_steps = 200,
    logging_steps = 50,
    logging_dir='/home/jovyan/LLM-Stance-Labeling/adapter_model_tuning_logs',
    fp16=True # mixed-precision training
)

# Create a formatter for the data so it works with SFT
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['context'])):
        if example['dataset'][i] == 'semeval':
            text = f"{example['context'][i]}\n entity: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
        elif example['dataset'][i] == 'election':
            text = f"{example['context'][i]}\n politcian: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
        elif example['dataset'][i] == 'covid':
            text = f"{example['context'][i]}\n belief: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
        elif example['dataset'][i] == 'pheme':
            text = f"{example['context'][i]}\n rumor: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
        elif example['dataset'][i] == 'wt':
            text = f"{example['context'][i]}\n merger: {example['event'][i]}\n statement: {example['full_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
        elif example['dataset'][i] == 'srq':
            text = f"{example['context'][i]}\n social media post: {example['target_text'][i]}\n topic: {example['event'][i]}\n statement: {example['response_text'][i]}\n</s> <s> stance: {example['train_stance'][i]}"
        output_texts.append(text)
    return output_texts  
  
response_template = "stance:"  
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

# Set up peft for efficient tuning
peft_config = LoraConfig(  
    r=256,  
    lora_alpha=16,  
    lora_dropout=0.1,  
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
)  

# create the SFT trainer
trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset, 
    formatting_func=formatting_prompts_func,  
    data_collator=collator,  
    peft_config=peft_config,
    max_seq_length = 650
)  

print(f"Is the model parallelized across devices: {trainer.is_model_parallel}")
print(f"Is FSDP enabled for training: {trainer.is_fsdp_enabled}")


trainer.train()

trainer.save_model("/home/jovyan/LLM-Stance-Labeling/stance_tuned_adapter_model")
