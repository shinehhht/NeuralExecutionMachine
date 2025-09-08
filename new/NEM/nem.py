import torch
import torch.nn as nn
import torch.nn.functional as F
from NEM.crossAttention import Transformer2Prog, Prog2Transformer
from NEM.featurePproj import FeatureProj
from NEM.interpreter import Interpreter
from types import SimpleNamespace


config = SimpleNamespace(
   hidden_dim = 1024,
   num_heads = 8,
   dropout = 0.3,
   total_slots = 34,
   num_instructions = 4,
   num_op = 2,
   num_cycles = 2,
   instruction_types = 4
)

class NeuralExecutionModule(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.Trans2Prog = Transformer2Prog(config)
        self.FeatureProj = FeatureProj(config)
        self.Interpreter = Interpreter(config)
        self.Fuse2Main = Prog2Transformer(config)
        
    def forward(self, hidden_states):
        
        opcode, op, mask, cond = self.Trans2Prog(hidden_states)

        updated_base = hidden_states               
        prog_tokens = []                             
        exec_outputs_list = []
        
        for cycle in range(self.config.num_cycles):
            context = (torch.cat([updated_base] + prog_tokens, dim=1)if prog_tokens else updated_base) 
               
            if cycle == 0:
                opcode_probs, mask_distribution, cond_distribution = self. FeatureProj(opcode, mask, cond)
                
            else:
                opcode, op, mask, cond = self.Trans2Prog(context)
                opcode_probs, mask_distribution, cond_distribution = self. FeatureProj(opcode, mask, cond)
            
            result,prodecure = self.Interpreter(opcode_probs,mask_distribution,cond_distribution,op)

            exec_outputs_list.append(result)
            
            added_result = result.unsqueeze(1)
            reftext = torch.cat([opcode, added_result], dim=1).contiguous()  # (b,k,d)
            prog_tokens.append(reftext)  
            
        exec_outputs = torch.stack(exec_outputs_list, dim=1)
        fused_hidden_states = self.Fuse2Main(hidden_states, exec_outputs)
        
        return fused_hidden_states
                
                
                
if __name__ == '__main__':
    hidden_states = torch.randn(2, 10, 1024, dtype=torch.float32)
    nem = NeuralExecutionModule(config)
    nem(hidden_states)