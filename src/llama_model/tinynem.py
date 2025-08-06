import torch
import torch.nn as nn
import torch.nn.functional as F
from llama_model.free_gradient_augmentation_head import Augmentation_Layer
from llama_model.free_gradient_utils import GroupSample
from utils import R_bit2decimal, bit2decimal, program_bit2decimal
from isa.isa_interpreter import MultiInterpreter, Interpreter

class TinyNEM(nn.Module):
    def __init__(self, hidden_dim, intermidate_dim, n_regs, n_val, prog_max_length, dropout):
        super().__init__()
        
        self.aug_layer = Augmentation_Layer(hidden_dim, intermidate_dim, n_regs, n_val, prog_max_length, dropout)
        self.groupSample = GroupSample(max_prog_len=16,op_bit=4, dst_bit=3, src1_bit=3, src2_bit=3, imm_bit=3, n_regs=8, r_bit=16, prog_len_bit=4)
        # self.groupSample = FieldsSample(max_prog_len=16,op_bit=4, dst_bit=3, src1_bit=3, src2_bit=3, imm_bit=3, n_regs=8, r_bit=16, prog_len_bit=4) 
        
        self.register2hidden = nn.Linear(n_regs, hidden_dim)
        nn.init.xavier_uniform_(self.register2hidden.weight, gain=1.0)
        nn.init.constant_(self.register2hidden.bias, 0.0)
        
        self.regLayernorm = nn.LayerNorm(hidden_dim, eps=1e-5)
        
    def forward(self, z_hidden, P, Proglen, group_num=10):
        
        Plogits, R, prog_len, threshold = self.aug_layer(z_hidden)
        current_set = {
            "R":R,
            "logits":P,
            "prog_len":Proglen
        }
        
        ##### group sample strategy
        candidate_set = self.groupSample.generate4R(current_set,group_num, flip_num=2)
        
        
        
        R_bit = torch.stack([c['R'] for c in candidate_set], dim=2)        # (B, L, group_num, n_regs, 16)
        logits_bit = { k: torch.stack([c['logits'][k] for c in candidate_set], dim=2) for k in candidate_set[0]['logits'].keys()}
        prog_len_bit = torch.stack([c['prog_len'] for c in candidate_set], dim=2) # (B,L,group_num,4)
        # print(f"op shape {logits_bit['op'].shape}")
        # print(f"prog shape {prog_len_bit.shape}")
        R = R_bit2decimal(R_bit) # (B,L,group,n_regs)
        M = torch.zeros_like(R)
        prog_len = bit2decimal(prog_len_bit) # (B,L,group)
        P = program_bit2decimal(logits_bit) # (B,L,group,max_length)
        
        interpreter = MultiInterpreter(R, M, P, prog_len)
        registers,_ = interpreter.run() # (B,L,group,n_regs)
        reg_hidden = self.register2hidden(registers) # (B,L,group,hidden_d)
        #print("Registers Before LayerNorm: min", reg_hidden.min().item(), "max", reg_hidden.max().item(), "std", reg_hidden.std().item())
        
        reg_hidden = self.regLayernorm(reg_hidden)
        
        return {
            'reg_hidden':reg_hidden, 
            'threshold':threshold,
            'candidate_set':candidate_set,
            'current_set':current_set
            } 
    
    def generate(self, z_hidden, P, Proglen):
        Plogits, R1, prog_len, threshold = self.aug_layer(z_hidden)
        
        R = (R1 >= 0.5).int()
        logits = {k: (v >=0.5).int() for k, v in P.items()}
        prolen = (Proglen >= 0.5).int()
        
        R_decimal = R_bit2decimal(R) # (B,L,n_regs)
        M = torch.zeros_like(R)
        prog_len_decimal = bit2decimal(prolen) # (B,L)
        P_decimal = program_bit2decimal(logits) # (B,L,max_length)
        # print(f"P shape is {P.shape}")
        interpreter = Interpreter(R_decimal, M, P_decimal, prog_len_decimal)
        registers,_ = interpreter.run() # (B,L,n_regs)
        
        reg_hidden = self.register2hidden(registers) # (B,L,hidden_d)
    
        reg_hidden = self.regLayernorm(reg_hidden)
        
        program = {
            'R':R_decimal,
            'P':P_decimal,
            'prog_len':prog_len_decimal
        }
        
        current_set={
            'R':R1,
            'logits':Plogits,
            'prog_len':prog_len
        }
         
        return reg_hidden, threshold, program, current_set
    
    