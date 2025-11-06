import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProj(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        self.opcode_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.instruction_types)  
        )
        
        self.mask_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)  
        
        )
        
        # 条件投影头
        self.cond_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1) 
        )
        
        
        
    def forward(self, opcode, mask, cond):
        instructions_distribution = F.softmax(self.opcode_head(opcode), dim=-1)
        mask_distribution = torch.sigmoid(self.mask_head(mask))
        
        #TODO
        cond_distribution = self.cond_head(cond)
        
        return instructions_distribution, mask_distribution, cond_distribution
    
class FeatureProjWithRegisters(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        
        self.opcode_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.instruction_types)  
        )
        
        """
        self.cond_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1) 
        )
        """
        
        self.gate_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1) 
        )

            
    def forward(self, opcode,gate):
        instructions_distribution = F.softmax(self.opcode_head(opcode), dim=-1)
        gate = torch.sigmoid(self.gate_head(gate))
        
        #TODO
        # cond_distribution = self.cond_head(cond)
        
        return instructions_distribution,  gate # cond_distribution, gate
        