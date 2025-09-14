import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split


class AddDataset(Dataset):
    def __init__(self, N, low, high, seed):
        g = torch.Generator().manual_seed(seed)
        
        a = torch.randint(low, high, (N, 1), generator=g)
        b = torch.randint(low, high, (N, 1), generator=g)
        x = torch.cat([a, b], dim=1)
        y = a + b
        
        self.x, self.y = x,y
        
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class AddMultibitDataset(Dataset):
    def __init__(self, file_path):
        dataset_dict = torch.load(file_path)
        
        self.data = dataset_dict['data']
        self.digits = dataset_dict['digits']
        self.max_input = dataset_dict['max_input']
        self.max_output = dataset_dict['max_output']
        self.out_digits = dataset_dict['out_digits']
        self.vocab_size = dataset_dict['vocab_size']
        self.num_samples = dataset_dict['num_samples']
        
        
    def __len__(self):
        return self.num_samples
    
    def to_bits(self, num, length):
        s = str(num).zfill(length)
        return [int(c) for c in s]
    
    def __getitem__(self, idx):
        a, b, c = self.data[idx]  

        a_digits = self.to_bits(a,self.digits)
        b_digits = self.to_bits(b,self.digits)
        c_digits = self.to_bits(c,self.out_digits)
        
        input_digits = a_digits + b_digits
        output_digits = c_digits

        x = torch.tensor(input_digits, dtype=torch.long)
        y = torch.tensor(output_digits, dtype=torch.long)
        
        return x, y
        
class MultiDataset(Dataset):
    def __init__(self, N, low, high, seed):
        g = torch.Generator().manual_seed(seed)
        per_num = N // 3
        
        def sample_ab(n):
            a = torch.randint(low, high, (n, 1), generator=g)
            b = torch.randint(low, high, (n, 1), generator=g)
            return a, b
        
        a_add, b_add = sample_ab(per_num)
        x_add = torch.cat([a_add, b_add], dim=1)
        y_add = (a_add + b_add)

       
        a_sub, b_sub = sample_ab(per_num)
        a_sub_sorted, b_sub_sorted = torch.maximum(a_sub, b_sub), torch.minimum(a_sub, b_sub)
        x_sub = torch.cat([a_sub_sorted, b_sub_sorted], dim=1)
        y_sub = (a_sub_sorted - b_sub_sorted)
       
      
        a_mul, b_mul = sample_ab(per_num)
        x_mul = torch.cat([a_mul, b_mul], dim=1)
        y_mul = ((a_mul * b_mul)% (2**15))
        
        x = torch.cat([x_add, x_sub, x_mul], dim=0)
        y = torch.cat([y_add, y_sub, y_mul], dim=0)   
        
        perm = torch.randperm(x.size(0), generator=g)
        x, y = x[perm], y[perm]         
        self.x, self.y = x, y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LoadAddDataset(Dataset):
    def __init__(self, x, y):                       
        self.x, self.y = x, y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] 
    
if __name__ == '__main__':
    dataset_token = AddMultibitDataset(N=5, digits=2)
    print(dataset_token)
    for x, y in dataset_token:
        print("token input:", x, " shape:", x.shape)
        print("token output:", y, " shape:", y.shape)