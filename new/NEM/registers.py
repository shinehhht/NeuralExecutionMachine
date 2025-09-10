import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Registers(nn.Module):
    def __init__(self, config, batch_size):
        super().__init__()
        self.config = config
        self.registers = torch.zeros(batch_size,config.n_regs,config.hidden_dim)
        
        
    def read(self,query):
        """
        key (B,2,d)
        """
        self.registers = self.registers.to(query.device)
        score = (query @ self.registers.transpose(1,2)) / math.sqrt(query.size(-1))
        w_read = F.softmax(score,dim=-1) #(B,2,n_regs)
        
        v_read = w_read @ self.registers #(B,2,d)
        
        return v_read
        
        
    def write(self,value, key,gate=None):
        """
        key (B,1,d)
        value (B,1,d)
        """
        self.registers = self.registers.to(key.device)
        score = (self.registers @ key.transpose(1,2)) / math.sqrt(key.size(-1))
        w_write = F.softmax(score,dim=1) #(B,n_regs,1)
        added = w_write @ value # (B,n_regs,d)
        if gate is None:
            self.registers =added
        else:
            portion = w_write * gate
            self.registers = (1-portion) * self.registers + portion * added
        
            
            
        
        
        