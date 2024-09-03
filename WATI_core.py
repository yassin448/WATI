import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import torch.nn as nn
# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Check if CUDA is available and use all GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')

# If more than one GPU is available, use DataParallel
if n_gpus > 1:
    model = nn.DataParallel(model)

model.to(device)

# Define the datasets to use
datasets = [
    load_dataset("wikitext", "wikitext-2-raw-v1", split='train'),
    load_dataset("empathetic_dialogues", split='train'),
    load_dataset("squad", split='train')
]

# Function to preprocess and tokenize datasets
def preprocess_and_tokenize(dataset, percentage, seed):
    dataset = dataset.shuffle(seed=seed).select(range(int(len(dataset) * percentage)))

    def tokenize_batch(batch):
        text_fields = []

        if 'text' in batch:
            text_fields = batch['text']
        elif 'utterance' in batch:
            text_fields = batch['utterance']
        elif 'context' in batch and 'question' in batch and 'answers' in batch:
            text_fields = [f"Context: {c} Question: {q} Answer: {a['text'][0]}" 
                           for c, q, a in zip(batch['context'], batch['question'], batch['answers'])]
        else:
            raise ValueError("Unsupported dataset format or missing fields.")
        
        inputs = tokenizer(text_fields, padding="max_length", truncation=True, max_length=256)
        inputs["labels"] = inputs["input_ids"].copy()
        
        return inputs

    dataset = dataset.map(tokenize_batch, batched=True)
    return dataset

# Apply preprocessing to the datasets
tokenized_datasets = [
    preprocess_and_tokenize(dataset, 0.05, seed=42) for dataset in datasets
]

# Combine datasets
combined_dataset = concatenate_datasets(tokenized_datasets)
combined_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_steps=200,
    logging_steps=50,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    eval_dataset=combined_dataset,  # Assuming you want to use the same dataset for eval
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
