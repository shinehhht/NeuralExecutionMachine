import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.tinymodel import MiniModel,ModelWithoutNEM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import config
import argparse
from torch.utils.tensorboard import SummaryWriter

class AddDataset(Dataset):
    def __init__(self, N, low, high, seed):
        g = torch.Generator().manual_seed(seed)
        
        a = torch.randint(low, high, (N, 1), generator=g)
        b = torch.randint(low, high, (N, 1), generator=g)
        x = torch.cat([a, b], dim=1).float()
        y = a + b
        
        self.x, self.y = x,y
        
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
class MultiDataset(Dataset):
    def __init__(self, N, low, high, seed):
        g = torch.Generator().manual_seed(seed)
        per_num = N // 3
        
        def sample_ab(n):
            a = torch.randint(low, high, (n, 1), generator=g)
            b = torch.randint(low, high, (n, 1), generator=g)
            return a, b
        
        a_add, b_add = sample_ab(per_num)
        x_add = torch.cat([a_add, b_add], dim=1).float()
        y_add = (a_add + b_add).float()

       
        a_sub, b_sub = sample_ab(per_num)
        x_sub = torch.cat([a_sub, b_sub], dim=1).float()
        y_sub = (a_sub - b_sub).float()
       
      
        a_mul, b_mul = sample_ab(per_num)
        x_mul = torch.cat([a_mul, b_mul], dim=1).float()
        y_mul = ((a_mul * b_mul)% (2**15)).float()
        
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
    
 
total = AddDataset(N=100000, low=0, high=10, seed=42)
torch.save({'x': total.x, 'y': total.y}, './training/simpleTransformer_test/add_dataset_classify.pt')

#data = torch.load('./training/simpleTransformer_test/add_dataset.pt')
#total = LoadAddDataset(data['x'], data['y'])

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


def train(args):
    file_name = args.file
    epochs = args.epochs
    writer = SummaryWriter(log_dir=f'./training/{file_name}')
    
    batch_size = 512
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size)

    
    model_without_NEM = ModelWithoutNEM(config).to(device)
    model_with_NEM =MiniModel(config).to(device)
    
    model_with_NEM_state_dict = model_with_NEM.state_dict()
    model_without_NEM_state_dict = model_without_NEM.state_dict()

   
    for name, param in model_without_NEM_state_dict.items():
        if name in model_with_NEM_state_dict and 'NEM' not in name:
            model_with_NEM_state_dict[name] = param
            
    optimizer_with_NEM = optim.Adam(model_with_NEM.parameters(), lr=0.00015)
    optimizer_without_NEM = optim.Adam(model_without_NEM.parameters(), lr=0.00015)
    loss_fn = torch.nn.CrossEntropyLoss() 

    ##### Train
    for epoch in tqdm(range(epochs)):
        model_with_NEM.train()
        model_without_NEM.train()
        
        total_loss1 = 0
        total_loss2 = 0
        num_samples = 0
        
        for inputs, labels in train_loader:

            inputs = inputs.unsqueeze(-1)
            labels = labels.squeeze(1)
            
            batch_size = inputs.size(0)
        
            inputs  = inputs.to(device, non_blocking=True)  
            labels  = labels.to(device, non_blocking=True)   

                
            logits = model_with_NEM(inputs) # output (B,19)
            loss = loss_fn(logits,labels)
            # loss = F.mse_loss(outputs, labels)
            total_loss1 += loss.item() * batch_size
            num_samples += batch_size
            optimizer_with_NEM.zero_grad()
            loss.backward()
            optimizer_with_NEM.step()


            logits_without_NEM = model_without_NEM(inputs)
            loss_without_NEM = loss_fn(logits_without_NEM, labels)
            total_loss2 += loss_without_NEM.item() * batch_size
            optimizer_without_NEM.zero_grad()
            loss_without_NEM.backward()
            optimizer_without_NEM.step()
            
            
        
            
        epoch_avg_loss1 = total_loss1 / max(1, num_samples)
        epoch_avg_loss2 = total_loss2 / max(1, num_samples)
        
        writer.add_scalar('Training Loss with NEM', epoch_avg_loss1, epoch)
        writer.add_scalar('Training Loss without NEM', epoch_avg_loss2, epoch)
        if (epoch+1)%10 == 0:
            with open(f'./training/{file_name}.txt','a')as f:
                f.write(f"Epoch [{epoch+1}/{epochs}], train_loss_with_NEM: {epoch_avg_loss1:.4f}\n")
                f.write(f"Epoch [{epoch+1}/{epochs}], train_loss_without_NEM: {epoch_avg_loss2:.4f}\n\n")
                
            
            
            
        ######  model validate
        model_with_NEM.eval()
        model_without_NEM.eval()
        
        val_loss1 = 0.0
        val_loss2 = 0.0
        val_num_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(-1)
                labels = labels.squeeze(1)
                
                batch_size = inputs.size(0)
                
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)


                logits1 = model_with_NEM.forward(inputs)
                logits2 = model_without_NEM.forward(inputs)
                loss1 = loss_fn(logits1, labels)
                loss2 = loss_fn(logits2, labels)
                val_loss1 += loss1.item() * batch_size
                val_loss2 += loss2.item() * batch_size
                val_num_samples += batch_size
        
        epoch_avg_val_loss1 = val_loss1 / max(1, val_num_samples)   
        epoch_avg_val_loss2 = val_loss2 / max(1, val_num_samples)   
        
        writer.add_scalar('Validation Loss with NEM', epoch_avg_val_loss1, epoch)
        writer.add_scalar('Validation Loss without NEM', epoch_avg_val_loss2, epoch)
        
        if (epoch+1)%10 == 0:
            with open(f'./training/{file_name}.txt','a')as f:
                f.write(f"Epoch [{epoch+1}/{epochs}], val_loss_with_NEM: {epoch_avg_val_loss1:.4f}\n")
                f.write(f"Epoch [{epoch+1}/{epochs}], val_loss_without_NEM: {epoch_avg_val_loss2:.4f}\n\n\n\n")
    
    torch.save(model_with_NEM.state_dict(), f"./training/simpleTransformer_test/nem_model_{file_name}.pth")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")

    parser.add_argument("--file", "-f", type=str, required=True, help=" ")
    parser.add_argument("--epochs", "-n", type=int, required=True, help=" ")
    args = parser.parse_args()
    
    train(args)