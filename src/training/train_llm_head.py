import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer, TrainingArguments, TrainerCallback
from llama_model.model import Augmented_LLM
from config import config
import torch
from datasets import load_from_disk, Dataset, DatasetDict

model_path = "/vepfs-mlp/mlp-public/zihaoyang/LLaMA/tinyllama"

LLM = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
LLM.resize_token_embeddings(len(tokenizer))
# tokenizer.pad_token_id = tokenizer.eos_token_id
#print(f"pad token {tokenizer.pad_token_id} {tokenizer.pad_token}")
#print(f"end token {tokenizer.eos_token_id} {tokenizer.eos_token}")
model = Augmented_LLM(LLM, tokenizer, config.n_regs, config.n_val, config.prog_max_length, config.dropout, LLM.config.hidden_size, config.max_tokens, config.intermidate_dim).to('cuda:0')

for name, param in model.named_parameters():
    if 'lm_head' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

  

def tokenize(line):
    if line["input"].strip():
        prompt = f"### Instruction:\n{line['instruction']}\n\n### Input:\n{line['input']}\n\n### Output:\n"
    else:
        prompt = f"### Instruction:\n{line['instruction']}\n\n### Output:\n"
    
    response = line["output"].strip() + tokenizer.eos_token  

   
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    print(f" prompt is {prompt_ids}")
    print(f" response is {response_ids}")
    
    input_ids = []
    if tokenizer.bos_token:
        input_ids.append(tokenizer.bos_token_id)
    input_ids += prompt_ids + response_ids

    attention_mask = [1] * len(input_ids)

    labels = [-100] * (len(prompt_ids) + (1 if tokenizer.bos_token else 0)) + response_ids

    if len(input_ids) > 512:
        input_ids = input_ids[:512]
        attention_mask = attention_mask[:512]
        labels = labels[:512]
    else:
        pad_len = 512 - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }   


class PeriodicInferenceCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_log(self, args, state, control, model=None, **kwargs):
        prompt = "### Instruction:\nWhat does 5 add 3 equal?\n\n### Output:\n"
        output = model.generate(prompt)
        dir_path = Path(__file__).parent
        record_path = dir_path / "record_parse.txt"
        with open(record_path, 'a')as f:
            f.write(f"\n{state.global_step//3} Inference:")
            f.write("\n\n")
            f.write(output)
            f.write("\n\n")
        
         
dataset = load_from_disk("/vepfs-mlp/mlp-public/zihaoyang/LLaMA/alpaca-cleaned")
testset = dataset["train"].select(range(1))
testset = DatasetDict({
    "train": testset  
})

tokenized_dataset = testset['train'].map(tokenize, remove_columns=testset["train"].column_names)           
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# print(tokenized_dataset[0])

training_args = TrainingArguments(
    output_dir="./demo",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_steps=3,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,        
    weight_decay=0.0,
    logging_strategy="steps", 
)

#print(f"train dataset is {len(tokenized_dataset)}")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    callbacks = [PeriodicInferenceCallback(tokenizer)]
)

trainer.train()
trainer.save_model("demo")
tokenizer.save_pretrained("demo")