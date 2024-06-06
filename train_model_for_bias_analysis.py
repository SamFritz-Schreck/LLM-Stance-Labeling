# pip install transformers datasets accelerate

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback
from datasets import Dataset

# Load the dataset
df = pd.read_pickle("all_data_training.pkl")
df = df[~df['train_stance'].isna()]

def map_labels(label):
    if label.lower() in ['supports', 'for']:
        return 'for'
    elif label.lower() in ['denies', 'against']:
        return 'against'
    elif label.lower() == 'neutral':
        return 'neutral'
    else:
        return label  # Return the label as is for unknown values

# Apply the mapping function to the 'train_stance' column
df['train_stance'] = df['train_stance'].map(map_labels)

labels = df['train_stance'].unique()
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# Define training arguments
training_args = TrainingArguments(
    output_dir='bias results/supervised_stance_checkpoints',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    weight_decay = 0.1,
    logging_dir='bias results/supervised_stance_model_logs',
    warmup_steps=100,
    logging_steps=20,
    save_steps=50,
    learning_rate= 3e-4,
    load_best_model_at_end=True,
    evaluation_strategy = IntervalStrategy.STEPS,
    eval_steps=50,
    #fp16=True, # mixed-precision training
)

# Create training and validation examples
train_data = []
for index, row in tqdm(df[~df['dataset'].isin(['election', 'semeval'])].iterrows(), total=len(df[~df['dataset'].isin(['election', 'semeval'])]), desc="Creating Training Examples"):
    if row['dataset'] == 'srq':
        example = f"target: {row['target_text']} [SEP] statement: {row['response_text']}"
        label = label2id[row['train_stance']]
    else:
        example = f"target: {row['event']} [SEP] statement: {row['full_text']}"
        label = label2id[row['train_stance']]
    train_data.append((example, label))

val_data = []
for index, row in tqdm(df[df['dataset'].isin(['election', 'semeval'])].iterrows(), total=len(df[df['dataset'].isin(['election', 'semeval'])]), desc="Creating Validation Examples"):
    if row['dataset'] == 'srq':
        example = f"target: {row['target_text']} [SEP] statement: {row['response_text']}"
        label = label2id[row['train_stance']]
    else:
        example = f"target: {row['event']} [SEP] statement: {row['full_text']}"
        label = label2id[row['train_stance']]
    val_data.append((example, label))

# Split off a validation dataset
# train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=True)

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
train_encodings = tokenizer([item[0] for item in train_data], truncation=True, padding=True)
val_encodings = tokenizer([item[0] for item in val_data], truncation=True, padding=True)

# Convert labels to numerical values
train_labels = [item[1] for item in train_data]
val_labels = [item[1] for item in val_data]

# Create Dataset objects
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels}).shuffle()
val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'], 'labels': val_labels})


# Create the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3, id2label=id2label, label2id=label2id)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=7)]
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model('bias results/supervised_stance_model')
