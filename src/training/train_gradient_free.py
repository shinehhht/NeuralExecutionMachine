import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer, TrainingArguments, TrainerCallback
from llama_model.free_gradient_model import Augmented_LLM
from config import config
import torch
from torch.optim import AdamW
from datasets import load_from_disk, Dataset, DatasetDict
from data.generate_basic_arithmetic import ArithmeticDataGenerator

def print_grad_hook(grad):
    print("Got gradient with norm:", grad)
    
model_path = "/vepfs-mlp/mlp-public/zihaoyang/LLaMA/tinyllama"

LLM = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
LLM.resize_token_embeddings(len(tokenizer))
# tokenizer.pad_token_id = tokenizer.eos_token_id
#print(f"pad token {tokenizer.pad_token_id} {tokenizer.pad_token}")
#print(f"end token {tokenizer.eos_token_id} {tokenizer.eos_token}")
model = Augmented_LLM(LLM, tokenizer, config.n_regs, config.n_val, config.prog_max_length, config.dropout, LLM.config.hidden_size, config.max_tokens, config.intermidate_dim).to('cuda:0')
# model.register2hidden.weight.register_hook(print_grad_hook)

for name, param in model.named_parameters():
    # print(name)
    if 'LLM' in name:
        if 'lm_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    else:
        param.requires_grad = True

        
def get_param_groups(model):
    high_lr = []
    mid_lr = []
    low_lr = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  

        if any(x in name for x in ["aug_layer", "regLayernorm", "register2hidden", "zLayernorm"]):
            high_lr.append(param)
        elif "lm_head" in name:
            mid_lr.append(param)
        else:
            low_lr.append(param)

    return [
        {"params": high_lr, "lr": 1e-3},
        {"params": mid_lr,  "lr": 5e-5},
        {"params": low_lr,  "lr": 5e-5},
    ]
     

def tokenize(line):
    """
    if line["input"].strip():
        prompt = f"### Instruction:\n{line['instruction']}\n\n### Input:\n{line['input']}\n\n### Output:\n"
    else:
        prompt = f"### Instruction:\n{line['instruction']}\n\n### Output:\n"
    """
    prompt = f"### Input:\n{line['input']}\n\n### Output:\n"
    response = line["output"].strip() + tokenizer.eos_token  

   
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    
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


class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = get_param_groups(self.model)
            self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer
        
"""       
dataset = load_from_disk("/vepfs-mlp/mlp-public/zihaoyang/LLaMA/alpaca-cleaned")

testset = dataset["train"].select(range(10000))
testset = DatasetDict({
    "train": testset  
})


tokenized_dataset = dataset['train'].map(tokenize, remove_columns=dataset["train"].column_names)           
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# print(tokenized_dataset[0])
"""
data_generator = ArithmeticDataGenerator(50000)
dataset = data_generator.load()
tokenized_dataset = dataset['train'].map(tokenize, remove_columns=dataset["train"].column_names)           
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print(tokenized_dataset[0])

training_args = TrainingArguments(
    output_dir="./gradient_free_checkpoint",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    weight_decay=0.0,
    logging_strategy="steps", 
    #ddp_backend="nccl",
    #ddp_find_unused_parameters=False,
    #bf16=True,     
)

#print(f"train dataset is {len(tokenized_dataset)}")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
    #callbacks = [PrintGradCallback]
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model("gradient_free")
    tokenizer.save_pretrained("gradient_free_tokenizer")