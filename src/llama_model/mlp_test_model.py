import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from llama_model.nem import NeuralExecutionModule
from llama_model.tinynem import TinyNEM


class MiniModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.NEM = NeuralExecutionModule(config)
        self.fc1 = nn.Linear(2,config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim,1)
        
        
    def forward(self,x,portions=None,group_num=10,sample_mode=None):
        z_hidden = torch.tanh(self.fc1(x)).unsqueeze(1)
        results = self.NEM(z_hidden,portions,group_num,sample_mode)
        
        threshold = results['threshold']
        reg_hidden = results['reg_hidden']
        candidate_set = results['candidate_set']
        current_set = results['current_set']
        
        
        final_hidden = (1-threshold.unsqueeze(-1))* z_hidden.unsqueeze(2) + threshold.unsqueeze(-1) * reg_hidden
        outputs = self.fc2(final_hidden) # (B,L,G,1)
        return outputs, candidate_set, current_set
    
    def generate(self,x):
        z_hidden = torch.tanh(self.fc1(x)).unsqueeze(1) # (B,L,64)
        
        reg_hidden, threshold, program, current_set = self.NEM.generate(z_hidden)
        final_hidden = (1-threshold)* z_hidden + threshold * reg_hidden
        output = self.fc2(final_hidden) # (B,1,1)
        
        return output,program, current_set
        
        
class RegulateRegisterModel(nn.Module):
    def __init__(self,n_regs, n_val, prog_max_length, dropout, hidden_dim, intermidate_dim):
        super().__init__()
        
        self.NEM = TinyNEM(hidden_dim,intermidate_dim,n_regs,n_val,prog_max_length,dropout)
        self.fc1 = nn.Linear(2,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
        
        
    def forward(self,x, P, proglen):
        z_hidden = torch.tanh(self.fc1(x)).unsqueeze(1)
        results = self.NEM(z_hidden, P, proglen)
        
        threshold = results['threshold']
        reg_hidden = results['reg_hidden']
        candidate_set = results['candidate_set']
        current_set = results['current_set']
        
        
        final_hidden = (1-threshold.unsqueeze(-1))* z_hidden.unsqueeze(2) + threshold.unsqueeze(-1) * reg_hidden
        outputs = self.fc2(final_hidden) # (B,G,1)
        
        return outputs, candidate_set, current_set
    
    def generate(self,x, P, proglen):
        z_hidden = torch.tanh(self.fc1(x)).unsqueeze(1)
        reg_hidden, threshold, program, current_set= self.NEM.generate(z_hidden, P, proglen)
        
        final_hidden = (1-threshold)* z_hidden + threshold * reg_hidden
        output = self.fc2(final_hidden) # (B,1)
        
        return output,program,current_set
       
        