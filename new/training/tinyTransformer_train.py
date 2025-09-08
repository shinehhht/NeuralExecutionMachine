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
from utils import assistLoss, computportions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import config
import argparse
from torch.utils.tensorboard import SummaryWriter

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
    
 
#total = AddDataset(N=100000, low=0, high=10, seed=42)
#torch.save({'x': total.x, 'y': total.y}, './simpleTransformer_test/add_dataset.pt')

data = torch.load('./training/simpleTransformer_test/add_dataset.pt')
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
            print('yes')
            model_with_NEM_state_dict[name] = param
            
    optimizer_with_NEM = optim.Adam(model_with_NEM.parameters(), lr=0.001)
    optimizer_without_NEM = optim.Adam(model_without_NEM.parameters(), lr=0.001)


    ##### Train
    for epoch in tqdm(range(epochs)):
        model_with_NEM.train()
        model_without_NEM.train()
        
        total_loss1 = 0
        total_loss2 = 0
        num_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(-1)

            batch_size = inputs.size(0)
        
            inputs  = inputs.to(device, non_blocking=True)  
            labels  = labels.to(device, non_blocking=True)   

                
            outputs = model_with_NEM(inputs) # output (B,1)
            loss = F.mse_loss(outputs, labels)
            total_loss1 += loss.item() * batch_size
            num_samples += batch_size
            optimizer_with_NEM.zero_grad()
            loss.backward()
            optimizer_with_NEM.step()


            outputs_without_NEM = model_without_NEM(inputs)
            loss_without_NEM = F.mse_loss(outputs_without_NEM, labels)
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
                batch_size = inputs.size(0)
                inputs = inputs.to(device, non_blocking=True)

                labels = labels.to(device, non_blocking=True)


                outputs1 = model_with_NEM.forward(inputs)
                outputs2 = model_without_NEM.forward(inputs)
                loss1 = F.mse_loss(outputs1, labels)
                loss2 = F.mse_loss(outputs2, labels)
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