import torch
import random
import os

def generate_add_dataset(N, digits, save_path='addition_dataset.pt'):
   
    max_input = 10 ** digits
    max_output = 2 * (10 ** digits - 1)
    out_digits = len(str(max_output))
    
    data = []
    for i in range(N):
        a = random.randint(0, max_input - 1)
        b = random.randint(0, max_input - 1)
        c = a + b
        data.append((a, b, c))
        
    
    dataset_dict = {
        'data': data,
        'digits': digits,
        'max_input': max_input,
        'max_output': max_output,
        'out_digits': out_digits,
        'vocab_size': 10,
        'num_samples': N
    }
    
    torch.save(dataset_dict, save_path)
   
   
generate_add_dataset(N=10000, digits=3, save_path='./simpleTransformer_test/multibit_add_3bit_dataset_10k.pt')
