from transformers import AutoModelForCausalLM,AutoTokenizer,LlamaConfig
from llama_model.model import Augmented_LLM
from config import config
import torch
import argparse

def predict(mode = 'ft'):
    if mode == 'ft':
        model_path = "/vepfs-mlp/mlp-public/zihaoyang/LLaMA/src/training/lmhead-finetuned-1000/model.safetensors"
        tokenizer_path = "/vepfs-mlp/mlp-public/zihaoyang/LLaMA/src/training/lmhead-tokenizer-finetuned-1000"
        aug_LLM = Augmented_LLM.from_pretrained(model_path, tokenizer_path).to('cuda:0')
        input_text = "What does 5+3 equal to?"
        output = aug_LLM.generate(input_text)

    elif mode == 'pretrain':
        model_path = "/vepfs-mlp/mlp-public/zihaoyang/LLaMA/tinyllama"
        LLM = AutoModelForCausalLM.from_pretrained(model_path).to('cuda:1')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        aug_LLM = Augmented_LLM(LLM, tokenizer, config.n_regs, config.n_val, config.prog_max_length, config.dropout, LLM.config.hidden_size, config.max_tokens, config.intermidate_dim).to('cuda:0')
        output = aug_LLM.generate(input_text)
        
    print(f"output is\n\n {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="choose mode")

    parser.add_argument("--mode", "-i", type=str, required=True, help="mode type")
    
    args = parser.parse_args()
    
    predict(args.mode)

