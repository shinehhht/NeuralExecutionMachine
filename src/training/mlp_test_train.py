import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_model.mlp_test_model import MiniModel
from utils import assistLoss, computportions, update_next_portions
from isa.isa_parse import Parser
from config import config
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AddDataset(Dataset):
    def __init__(self, N, low=0, high=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        a = torch.randint(low, high, (N, 1), generator=g)
        b = torch.randint(low, high, (N, 1), generator=g)
        x = torch.cat([a.float(), b.float()], dim=1)  
        y = (a + b).float()                           
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
    
"""  
total = AddDataset(N=100000, low=0, high=10, seed=42)
torch.save({'x': total.x, 'y': total.y}, './mlp-test/add_dataset.pt')
"""

data = torch.load('./training/mlp-test/add_dataset.pt')
total = LoadAddDataset(data['x'], data['y'])

train_len = int(0.7 * len(total))
val_len = len(total) - train_len
train_set, val_set = random_split(total, [train_len, val_len],generator=torch.Generator().manual_seed(1234))

 
def make_loaders(train_set, val_set, batch_size, num_workers=4):
    train_sampler = val_sampler = None
    shuffle_train = True
    shuffle_val   = False

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle_train,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle_val,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def train(group_num,sample_mode,file_name):
    epochs = 100
    batch_size = 256
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size)

    model = MiniModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    portions = {
            'op':0.14,
            'dst':0.14,
            'src1':0.14,
            'src2':0.14,
            'imm':0.14,
            'R':0.14,
            'prog_len':0.14
        }
    alpha = 0.8

    ###### prepare actual set

    total_bit = config.n_op + config.n_dst + config.n_src1 + config.n_src2 + config.n_imm
    
    field_bits = {
    'op': config.n_op,
    'dst': config.n_dst,
    'src1': config.n_src1,
    'src2': config.n_src2,
    'imm': config.n_imm,
}
    instruction = ["add r0 r1 r2 0"]
    parser = Parser(config)
    P = parser.risc2binary(instruction) # (1,1,len)
    P = P.unsqueeze(-1)
    P = ((P >> torch.arange(total_bit-1, -1, -1)) & 1).int().to(device)
    
    logits = {}
    offset = 0
    for field, width in field_bits.items():
        logits[field] = P[..., offset : offset + width]
        offset += width
    
    proglen = torch.zeros(1,1,config.pc_bit, dtype=torch.int32).to(device)  #(1,1,2)
    proglen[..., -1] = 1


    for epoch in tqdm(range(epochs)):
        model.train()
        field_percentage = []
        actual_field_loss_list = []
        actual_total_loss = 0.0
        train_total_loss = 0.0
        total_samples = 0
        for inputs, labels in train_loader:
            batch_size = inputs.size(0)
            inputs  = inputs.to(device, non_blocking=True)  
            # print(f"input shape is {inputs.shape}") 
            labels  = labels.to(device, non_blocking=True)   

            R_actual = parser.assignRegisterMlp(inputs)
            actual_target = {
                    'R':R_actual.to(device),
                    'logits':logits,
                    'prog_len':proglen
                }
                
            outputs, candidate_set, current_set = model(inputs,portions,group_num,sample_mode) # output (B,1,G,1)
            labels = labels.unsqueeze(-1).expand(-1,1, len(candidate_set))
            
            loss = F.mse_loss(outputs.squeeze(-1), labels, reduction='none')
            loss = loss.sum(dim=1).sum(0)

            best_idx = torch.argmin(loss,dim=-1).item()
            #print(f"best idx is {best_idx}, loss is {loss[best_idx]}")
            best_candidate =  candidate_set[best_idx]
            
            
            assist_loss, field_loss = assistLoss(current_set, actual_target)
            field_percentage.append(field_loss)
            
            final_loss = loss[0]
            total_loss = 0.5 * final_loss + 0.5 * assist_loss
            train_total_loss += final_loss
            total_samples += batch_size
            
            ###### testing for actual distribution
            actual_loss, actual_field_loss = assistLoss(current_set, actual_target)
            actual_total_loss += actual_loss
            actual_field_loss_list.append(actual_field_loss)
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        
        field_loss_bit, cur_portions = computportions(field_percentage)
        portions = update_next_portions(alpha, cur_portions, portions)
        
        
        actual_program_loss_final = actual_total_loss / total_samples
        actual_field_loss_bit,_ = computportions(actual_field_loss_list)
            
        with open(f'./training/{file_name}.txt','a')as f:
            f.write(f"Epoch {epoch}: \nfield loss against 'Gtarget'= {{ {', '.join(f'{k}: {v.detach().item():.8f}' for k, v in field_loss_bit.items())} }} | average loss percentage = {{ {', '.join(f'{k}: {v.detach().item():.8f}' for k, v in portions.items())} }}\n")
            f.write(f"field loss against 'Atarget' = {{ {', '.join(f'{k}: {v.detach().item():.8f}' for k, v in actual_field_loss_bit.items())} }} | actual program_loss :{actual_program_loss_final:.8f}\n\n")
        
        if (epoch+1)%10 == 0:
            with open(f'./training/{file_name}.txt','a')as f:
                f.write(f"Epoch [{epoch+1}/{epochs}], train_loss: {(train_total_loss.item() / total_samples):.4f}\n")
            
        model.eval()
        val_loss = 0.0
        program_total_loss = 0.0
        with torch.no_grad():
            field_loss_list = []
            total_val_samples = 0
            for inputs, labels in val_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(device, non_blocking=True)

                labels = labels.to(device, non_blocking=True)

                R_actual = parser.assignRegisterMlp(inputs)
                actual_target = {
                        'R':R_actual.to(device),
                        'logits':logits,
                        'prog_len':proglen
                    }
                
                outputs, program, current_set = model.generate(inputs)
                # print(f"output shpae is {outputs.shape}")
                output = outputs.squeeze(-1) # (B,1)
                # print(f"output is {output}, shpae is {output.shape}")

                loss = F.mse_loss(output, labels)
                
                program_loss, field_loss = assistLoss(current_set, actual_target)
                program_total_loss += program_loss
                field_loss_list.append(field_loss)
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
                
            program_loss_final = program_total_loss / total_val_samples
            field_loss_bit,_ = computportions(field_loss_list)
            
        if (epoch+1)%10 == 0:
            with open(f'./training/{file_name}.txt','a')as f:
                f.write(f"\nEpoch {epoch+1}/{epochs} | val_loss: {(val_loss/total_val_samples):.4f} | program_loss :{program_loss_final:.8f} | field loss against 'Atarget' = {{ {', '.join(f'{k}: {v.detach().item():.8f}' for k, v in field_loss_bit.items())} }}\n\n" )
        
    torch.save(model.state_dict(), f"./training/mlp-test/mlp_model_{file_name}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")

    parser.add_argument("--file", "-f", type=str, required=True, help=" ")
    parser.add_argument("--group_num", "-g", type=int, required=True, help=" ")
    parser.add_argument("--mode", "-m", type=str, required=True, help=" ")
    args = parser.parse_args()
    
    train(args.group_num, args.mode, args.file)

    
    
    
    