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
from utils import compute_acc, count_correct, count_correct_multibits, grad_norm
from dataset import AddMultibitDataset
import argparse
from torch.utils.tensorboard import SummaryWriter


dataset = AddMultibitDataset('./training/simpleTransformer_test/multibit_add_3bit_dataset_1k.pt')

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len],generator=torch.Generator().manual_seed(42))


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
    file_name = args.file+'_'+str(args.lr)+'_'+str(args.betas)+'_'+str(args.weight_decay)

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
    
    gate_params = list(model_with_NEM.NEM.Gate.parameters())
    nem_params = [p for n,p in model_with_NEM.named_parameters() if "Gate" not in n]
   
    lr_gate = args.lr * 0.5
    lr_nem = args.lr * 5
    optimizer_with_NEM = optim.Adam(
        [{"params": gate_params, "lr": lr_gate},     
        {"params": nem_params,  "lr": lr_nem}, ],
        weight_decay = args.weight_decay,
        betas = args.betas
        )
    
    optimizer_without_NEM = optim.Adam(
        model_without_NEM.parameters(), 
        lr = args.lr,
        weight_decay = args.weight_decay,
        betas = args.betas
        )
    
    loss_fn = torch.nn.CrossEntropyLoss() 

    ##### Train
    for epoch in tqdm(range(epochs)):
        model_with_NEM.train()
        model_without_NEM.train()
        
        total_loss1 = 0
        total_loss2 = 0
        
        total_correct1 = 0
        total_correct2 = 0
        
        num_samples = 0
        
        for batch, (inputs, labels) in enumerate(train_loader):
            
            # labels = labels.squeeze(1)
            
            batch_size = inputs.size(0)
        
            inputs  = inputs.to(device, non_blocking=True)  
            labels  = labels.to(device, non_blocking=True)   

                
            logits = model_with_NEM(inputs,epoch,batch,file_name,writer).transpose(1,2) # output (B,10,bits)
            loss = loss_fn(logits,labels)
            
            total_correct1 += count_correct_multibits(labels,logits)
            total_loss1 += loss.item() * batch_size
            num_samples += batch_size
            optimizer_with_NEM.zero_grad()
            loss.backward()
            gn_gate,rms_gate = grad_norm(model_with_NEM.NEM.Gate.named_parameters(),except_gate=False)
            gn_total, rms_total = grad_norm(model_with_NEM.NEM.named_parameters(),except_gate=True)
            if batch == 0:
                writer.add_scalar('grad_norm/L2/Gate', gn_gate, epoch)
                writer.add_scalar('grad_norm/L2/NEM except Gate', gn_total, epoch)
                
                writer.add_scalars(
                    main_tag='grad_norm/RMS',
                    tag_scalar_dict={
                        'Gate':           rms_gate,
                        'NEM_except_Gate': rms_total
                    },
                    global_step=epoch
                )
                
            optimizer_with_NEM.step()


            logits_without_NEM = model_without_NEM(inputs).transpose(1,2)
            loss_without_NEM = loss_fn(logits_without_NEM, labels)
            
            total_correct2 += count_correct_multibits(labels,logits_without_NEM)
            total_loss2 += loss_without_NEM.item() * batch_size
            optimizer_without_NEM.zero_grad()
            loss_without_NEM.backward()
            optimizer_without_NEM.step()
            
            
        
            
        epoch_avg_loss1 = total_loss1 / max(1, num_samples)
        epoch_avg_loss2 = total_loss2 / max(1, num_samples)
        
        epoch_avg_acc1 = total_correct1 / max(1, num_samples)
        epoch_avg_acc2 = total_correct2 / max(1, num_samples)
        
        writer.add_scalar('Training Loss with NEM', epoch_avg_loss1, epoch)
        writer.add_scalar('Training Loss without NEM', epoch_avg_loss2, epoch)
        
        writer.add_scalar('Accuracy with NEM', epoch_avg_acc1, epoch)
        writer.add_scalar('Accuracy without NEM', epoch_avg_acc2, epoch)
        
        
        if (epoch+1)%10 == 0:
            with open(f'./training/{file_name}.txt','a')as f:
                f.write(f"Epoch [{epoch+1}/{epochs}], train_loss_with_NEM: {epoch_avg_loss1:.4f}, acc: {epoch_avg_acc1:.4f}\n")
                f.write(f"Epoch [{epoch+1}/{epochs}], train_loss_without_NEM: {epoch_avg_loss2:.4f}, acc: {epoch_avg_acc2:.4f}\n\n")
             
            
            
            
        ######  model validate
        model_with_NEM.eval()
        model_without_NEM.eval()
        
        val_loss1 = 0.0
        val_loss2 = 0.0
        
        val_total_correct1 = 0
        val_total_correct2 = 0
        
        val_num_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # labels = labels.squeeze(1)
                
                batch_size = inputs.size(0)
                
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)


                logits1 = model_with_NEM.generate(inputs).transpose(1,2)
                logits2 = model_without_NEM(inputs).transpose(1,2)
                
                loss1 = loss_fn(logits1, labels)
                loss2 = loss_fn(logits2, labels)
                
                val_total_correct1 += count_correct_multibits(labels,logits1)
                val_total_correct2 += count_correct_multibits(labels,logits2)
                
                val_loss1 += loss1.item() * batch_size
                val_loss2 += loss2.item() * batch_size
                val_num_samples += batch_size
        
        epoch_avg_val_loss1 = val_loss1 / max(1, val_num_samples)   
        epoch_avg_val_loss2 = val_loss2 / max(1, val_num_samples)   
        
        epoch_avg_val_acc1 = val_total_correct1 / max(1, val_num_samples)   
        epoch_avg_val_acc2 = val_total_correct2 / max(1, val_num_samples)   
        
        writer.add_scalar('Validation Loss with NEM', epoch_avg_val_loss1, epoch)
        writer.add_scalar('Validation Loss without NEM', epoch_avg_val_loss2, epoch)
        
        writer.add_scalar('Validation Accuracy with NEM', epoch_avg_val_acc1, epoch)
        writer.add_scalar('Validation Accuracy without NEM', epoch_avg_val_acc2, epoch)
        
        
        if (epoch+1)%10 == 0:
            with open(f'./training/{file_name}.txt','a')as f:
                f.write(f"Epoch [{epoch+1}/{epochs}], val_loss_with_NEM: {epoch_avg_val_loss1:.4f}, acc: {epoch_avg_val_acc1:.4f}\n")
                f.write(f"Epoch [{epoch+1}/{epochs}], val_loss_without_NEM: {epoch_avg_val_loss2:.4f}, acc: {epoch_avg_val_acc2:.4f}\n\n\n\n")
        
        
    torch.save(model_with_NEM.state_dict(), f"./training/simpleTransformer_test/nem_model_{file_name}.pth")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")

    parser.add_argument("--file", "-f", type=str, required=True, help=" ")
    parser.add_argument("--epochs", "-n", type=int, required=True, help=" ")
    parser.add_argument("--lr", "-l", type=float, required=True, help=" ")
    parser.add_argument("--weight_decay", "-w", type=float, required=True, help=" ")
    parser.add_argument("--betas", "-b", type=float, nargs=2, default=(0.9, 0.999))
    
    args = parser.parse_args()
    
    train(args)