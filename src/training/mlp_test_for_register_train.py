import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_model.mlp_test_model import RegulateRegisterModel
from utils import assistLoss4R
from isa.isa_parse import Parser

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

data = torch.load('./mlp-test/add_dataset.pt')
total = LoadAddDataset(data['x'], data['y'])

train_len = int(0.9 * len(total))
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


epochs = 50
batch_size = 256
train_loader, val_loader = make_loaders(train_set, val_set, batch_size)
parser = Parser()

instruction = ["add r1 r2 r3 0"]
P = parser.risc2binary(instruction) # (1,1,len)
P = P.repeat(batch_size,1,1).unsqueeze(-1)
P = ((P >> torch.arange(15, -1, -1)) & 1).int()  
Plogits = {
    'op':P[..., :4],
    'dst':P[...,4:7],
    'src1':P[...,7:10],
    'src2':P[...,10:13],
    'imm':P[...,13:16]
}
# print(f"P shape is {P.shape}")
proglen = torch.zeros(1,1,4, dtype=torch.int32)  #(B,L,4)
proglen[..., -1] = 1


model = RegulateRegisterModel(8,16,16,0.3,64,64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(epochs)):
    model.train()
    train_total_loss = 0.0
    total_samples = 0
    actual_total_loss = 0.0
    target_total_loss = 0.0
    field_percentage = []
    actual_field_loss_list = []
    
    for inputs, labels in train_loader:
        inputs  = inputs.to(device, non_blocking=True)   
        labels  = labels.to(device, non_blocking=True)   
        R_actual = parser.assignRegisterMlp(inputs)
        
        program_len = proglen.repeat(inputs.size(0),1,1)
        #print(f"proglen shape {proglen.shape}")
        outputs, candidate_set, current_set = model(inputs, Plogits, program_len) # output (B,1,G,1)
        labels = labels.unsqueeze(-1).expand(-1,1, len(candidate_set))
        
        loss = F.mse_loss(outputs.squeeze(-1), labels, reduction='none')
        loss = loss.sum(dim=1).sum(0)

        best_idx = torch.argmin(loss,dim=-1).item()
        # print(f"best idx is {best_idx}, loss is {loss[best_idx]}")
        best_candidate =  candidate_set[best_idx]
        assist_loss,R_loss = assistLoss4R(current_set, best_candidate)
        field_percentage.append(R_loss)
        
        final_loss = loss[0]
        target_total_loss += assist_loss
        total_loss = 0.8 * final_loss + 0.2 * assist_loss
        train_total_loss += final_loss
        total_samples += batch_size
        
        
        ###### testing for actual distribution
        actual_loss,_ = assistLoss4R(current_set, {'R':R_actual.to(device)})
        actual_total_loss += actual_loss
        
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    
    actual_R_loss = actual_total_loss / total_samples
    target_R_loss = target_total_loss / total_samples
    
    with open("groupsample_onlyfor_R.txt",'a')as f:
        f.write(f"Epoch {epoch}: \nR loss against 'Gtarget'= {target_R_loss}\n")
        f.write(f"R loss against 'Atarget' = {actual_R_loss}\n")
        
    if (epoch+1)%10 == 0:
        with open("groupsample_onlyfor_R.txt",'a')as f:
            f.write(f"Epoch [{epoch+1}/{epochs}], train_loss: {(train_total_loss.item()/total_samples):.4f}")
        
    model.eval()
    val_loss = 0.0
    program_total_loss = 0.0
    with torch.no_grad():
        total_val_samples = 0
        for inputs, labels in val_loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            R_actual = parser.assignRegisterMlp(inputs)
            outputs, program,current_set = model.generate(inputs,Plogits,proglen)
            #print(f"shpae is {outputs.shape}")
            output = outputs.squeeze(-1) # (B,1)
            

            loss = F.mse_loss(output, labels)
            print(f"loss is {loss}")
            actual_loss, _ = assistLoss4R(current_set, {'R':R_actual.to(device)})
            program_total_loss += actual_loss
            val_loss += loss.item() * batch_size
            total_val_samples += batch_size
        
        program_loss_final = program_total_loss / total_val_samples
            
    if (epoch+1)%10 == 0:
        with open("groupsample_onlyfor_R.txt",'a')as f:
            f.write(f"\nEpoch {epoch+1}/{epochs} | val_loss: {(val_loss/total_val_samples):.8f} | R_loss :{program_loss_final:.8f}\n\n" )
    
    
torch.save(model.state_dict(), "mlp_model4R.pth")




    
    
    
    