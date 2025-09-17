import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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

def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def cleanup():
    dist.destroy_process_group()
    

def make_loaders_ddp(train_set, val_set, batch_size, rank, world_size, num_workers=4):
    
    train_sampler = DistributedSampler(
        train_set, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    batch_size_per_gpu = batch_size // world_size
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size_per_gpu, 
        shuffle=False, 
        sampler=train_sampler, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size_per_gpu, 
        shuffle=False,
        sampler=val_sampler, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader, train_sampler, val_sampler


def train_ddp(args):
    rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{rank}')
    
    if args.constrain:
        file_name = 'clip_'+args.file+'_'+str(args.lr)+'_'+str(args.betas)+'_'+str(args.weight_decay)+'_'+str(args.T1)+'_'+str(args.T2)+'_'+str(args.U1)+'_'+str(args.L)+'_'+str(args.U2)
    else:
        file_name = args.file+'_'+str(args.lr)+'_'+str(args.betas)+'_'+str(args.weight_decay)
        
    epochs = args.epochs
    batch_size = args.batch_size
    
    writer=None
    if rank == 0:
        writer = SummaryWriter(log_dir=f'./training/log/{file_name}_ddp')
    
    dataset_file_path = './training/simpleTransformer_test/'+ args.dataset_file
    dataset = AddMultibitDataset(dataset_file_path)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len],generator=torch.Generator().manual_seed(42))
    
    train_loader, val_loader, train_sampler, val_sampler = make_loaders_ddp(
        train_set, val_set, batch_size, rank, world_size
    )
    
    model_without_NEM = ModelWithoutNEM(config,args).to(device)
    model_with_NEM = MiniModel(config,args).to(device)
    
    model_with_NEM_state_dict = model_with_NEM.state_dict()
    model_without_NEM_state_dict = model_without_NEM.state_dict()

   
    for name, param in model_without_NEM_state_dict.items():
        if name in model_with_NEM_state_dict and 'NEM' not in name:
            model_with_NEM_state_dict[name] = param
            

    
    if args.multi_decay:
        decay_param = [p for n,p in model_with_NEM.named_parameters() if "Gate" not in n]
        nodecay_param = list(model_with_NEM.NEM.Gate.parameters())
        
        param_groups = [
        {"params": decay_param, "weight_decay":args.weight_decay},
        {"params": nodecay_param, "weight_decay": 0.0},  
        ]
        
        optimizer_with_NEM = torch.optim.AdamW(
            param_groups, 
            lr = args.lr,
            betas = args.betas
            )
         
    elif args.multi_lr:
        gate_params = list(model_with_NEM.NEM.Gate.parameters())
        nem_params = [p for n,p in model_with_NEM.NEM.named_parameters() if "Gate" not in n]
        other_params = [p for n,p in model_with_NEM.named_parameters() if "NEM" not in n]
        
        lr_gate = args.lr * 0.1
        
        param_groups = [
        {"params": gate_params, "lr": lr_gate},
        {"params": nem_params,  "lr": args.lr},
        {"params": other_params, "lr": args.lr} 
        ]
         
        optimizer_with_NEM = optim.Adam(
            param_groups,
            weight_decay = args.weight_decay,
            betas = args.betas
            )
        
    else:
        optimizer_with_NEM = optim.Adam(
        model_with_NEM.parameters(), 
        lr = args.lr,
        weight_decay = args.weight_decay,
        betas = args.betas
        )
    
    optimizer_without_NEM = optim.Adam(
        model_without_NEM.parameters(), 
        lr = args.lr,
        weight_decay = args.weight_decay,
        betas = args.betas
        )

        
    model_with_NEM = DDP(model_with_NEM, device_ids=[rank], output_device=rank,find_unused_parameters=True)
    model_without_NEM = DDP(model_without_NEM, device_ids=[rank], output_device=rank,find_unused_parameters=True)

    
    loss_fn = torch.nn.CrossEntropyLoss() 

   
    ##### Train
    for epoch in tqdm(range(epochs)):
        train_sampler.set_epoch(epoch)
        
        model_with_NEM.train()
        model_without_NEM.train()
        
        total_loss1 = 0
        total_loss2 = 0
        
        total_correct1 = 0
        total_correct2 = 0
        
        num_samples = 0
        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch, (inputs, labels) in enumerate(train_loader):
            
            # labels = labels.squeeze(1)
            
            batch_size = inputs.size(0)
        
            inputs  = inputs.to(device, non_blocking=True)  
            labels  = labels.to(device, non_blocking=True)   

            if rank == 0:   
                logits,origin_gate,processed_gate = model_with_NEM(inputs,epoch) # output (B,10,bits)
                logits = logits.transpose(1,2)
            else:
                logits,_,_ = model_with_NEM(inputs,epoch) # output (B,10,bits)
                logits = logits.transpose(1,2)
            loss = loss_fn(logits,labels)
            
            total_correct1 += count_correct_multibits(labels,logits)
            total_loss1 += loss.item() * batch_size
            num_samples += batch_size
            optimizer_with_NEM.zero_grad()
            loss.backward()
            
            
            gn_gate,rms_gate = grad_norm(model_with_NEM.module.NEM.Gate.named_parameters(),except_gate=False)
            gn_total, rms_total = grad_norm(model_with_NEM.module.NEM.named_parameters(),except_gate=True)
            if rank==0 and batch == 0:
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
            
            if rank == 0:
                pbar.update(1)
        
        metrics = torch.tensor([total_loss1, total_loss2, total_correct1, total_correct2, num_samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        total_loss1 = metrics[0].item()
        total_loss2 = metrics[1].item()
        total_correct1 = metrics[2].item()
        total_correct2 = metrics[3].item()
        num_samples = metrics[4].item()
        
        epoch_avg_loss1 = total_loss1 / max(1, num_samples)
        epoch_avg_loss2 = total_loss2 / max(1, num_samples)
        
        epoch_avg_acc1 = total_correct1 / max(1, num_samples)
        epoch_avg_acc2 = total_correct2 / max(1, num_samples)
        
        if rank == 0:
            writer.add_scalar('TrainingLoss/Loss with NEM', epoch_avg_loss1, epoch)
            writer.add_scalar('TrainingLoss/Loss without NEM', epoch_avg_loss2, epoch)
            
            writer.add_scalar('TrainingAccuracy/Accuracy with NEM', epoch_avg_acc1, epoch)
            writer.add_scalar('TrainingAccuracy/Accuracy without NEM', epoch_avg_acc2, epoch)
            
            
            if args.constrain:
                writer.add_scalar('Gate/Origin gate', origin_gate, epoch)
                writer.add_scalar('Gate/Processed gate', processed_gate, epoch)
            else:
                writer.add_scalar('Gate', origin_gate, epoch)
            
            
            if (epoch+1)%5 == 0:
                with open(f'./training/record/{file_name}.txt','a')as f:
                    f.write(f"Epoch [{epoch+1}/{epochs}], train_loss_with_NEM: {epoch_avg_loss1:.4f}, acc: {epoch_avg_acc1:.4f}\n")
                    f.write(f"Epoch [{epoch+1}/{epochs}], train_loss_without_NEM: {epoch_avg_loss2:.4f}, acc: {epoch_avg_acc2:.4f}\n\n")
             
            
        if (epoch + 1) % 10 == 0:
            val_sampler.set_epoch(epoch)
            model_with_NEM.eval()
            model_without_NEM.eval()
            
            val_loss1 = 0.0
            val_loss2 = 0.0
            
            val_total_correct1 = 0
            val_total_correct2 = 0
            
            val_num_samples = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    
                    batch_size = inputs.size(0)
                    
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)


                    logits1 = model_with_NEM.module.generate(inputs).transpose(1,2)
                    logits2 = model_without_NEM(inputs).transpose(1,2)
                    
                    loss1 = loss_fn(logits1, labels)
                    loss2 = loss_fn(logits2, labels)
                    
                    val_total_correct1 += count_correct_multibits(labels,logits1)
                    val_total_correct2 += count_correct_multibits(labels,logits2)
                    
                    val_loss1 += loss1.item() * batch_size
                    val_loss2 += loss2.item() * batch_size
                    val_num_samples += batch_size
            
            metrics = torch.tensor([val_loss1, val_loss2, val_total_correct1, val_total_correct2, val_num_samples], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            
            epoch_avg_val_loss1 = metrics[0].item() / metrics[4].item()
            epoch_avg_val_loss2 = metrics[1].item() / metrics[4].item() 
            
            epoch_avg_val_acc1 = metrics[2].item() / metrics[4].item()
            epoch_avg_val_acc2 = metrics[3].item() / metrics[4].item()
            
            if rank == 0:
                writer.add_scalar('ValLoss/Loss with NEM', epoch_avg_val_loss1, epoch)
                writer.add_scalar('ValLoss/Loss without NEM', epoch_avg_val_loss2, epoch)
                
                writer.add_scalar('ValAccuracy/Accuracy with NEM', epoch_avg_val_acc1, epoch)
                writer.add_scalar('ValAccuracy/Accuracy without NEM', epoch_avg_val_acc2, epoch)
                
            
                if (epoch+1)%10 == 0:
                    with open(f'./training/record/{file_name}.txt','a')as f:
                        f.write(f"Epoch [{epoch+1}/{epochs}], val_loss_with_NEM: {epoch_avg_val_loss1:.4f}, acc: {epoch_avg_val_acc1:.4f}\n")
                        f.write(f"Epoch [{epoch+1}/{epochs}], val_loss_without_NEM: {epoch_avg_val_loss2:.4f}, acc: {epoch_avg_val_acc2:.4f}\n\n\n\n")
                        
                    
    
    if rank == 0 and writer is not None:
        writer.close()
    
    cleanup()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument("--lr", "-l", type=float, required=True, help=" ")
    parser.add_argument("--weight_decay", "-w", type=float, required=True, help=" ")
    parser.add_argument("--betas", "-b", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--constrain","-c",action="store_true",help="default false")
    parser.add_argument("--multi_decay",action="store_true",help="default false")
    parser.add_argument("--multi_lr",action="store_true",help="default false")
    parser.add_argument('--T1', type=int, default=30)
    parser.add_argument("--T2", type=int, default=300)
    parser.add_argument("--U1", "-u", type=float, required=True, help=" ")
    parser.add_argument("--L", type=float )
    parser.add_argument("--U2", "-u2", type=float, required=True, help=" ")
    parser.add_argument("--eps", type=float,default=1e-5)
    parser.add_argument("--gate_ema","-e",action="store_true",help="default false")
    parser.add_argument("--gate_decay", type=float)
    parser.add_argument("--gate_alpha", type=float)
    parser.add_argument('--input_bits', type=int)
    parser.add_argument('--output_bits', type=int)
    
    args = parser.parse_args()
    
    train_ddp(args)