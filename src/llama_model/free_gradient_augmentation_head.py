import torch
import torch.nn as nn
import torch.nn.functional as F


class Augmentation_Layer(nn.Module):
    def __init__(self, config):
        """
        hidden_dim: (B,d_model) layer output by LLM
        
        """
        super().__init__()
        self.d_hidden = config.hidden_dim 
        self.n_regs = config.n_regs
        self.n_val = config.n_val
        self.prog_max_length = config.prog_max_length
        self.dropout = config.dropout
        
        self.pcbit = config.pc_bit
        self.n_op = config.n_op
        self.n_dst = config.n_dst
        self.n_src1 = config.n_src1
        self.n_src2 = config.n_src2
        self.n_imm = config.n_imm
        
        self.config = config
        self.proj_prog_thres = nn.Linear(self.d_hidden,1)
        nn.init.constant_(self.proj_prog_thres.weight, 1e-3)  
        nn.init.constant_(self.proj_prog_thres.bias, -1.0)


        self.proj_R = nn.Linear(self.d_hidden, self.n_regs*self.n_val) # (B, L, n_regs*n_val)
        
        self.proj_op   = nn.Linear(self.d_hidden, self.prog_max_length * config.n_op)  
        self.proj_dst  = nn.Linear(self.d_hidden, self.prog_max_length *  config.n_dst)
        self.proj_src1 = nn.Linear(self.d_hidden, self.prog_max_length *  config.n_src1)
        self.proj_src2 = nn.Linear(self.d_hidden, self.prog_max_length *  config.n_src2)
        self.proj_imm  = nn.Linear(self.d_hidden, self.prog_max_length *  config.n_imm)
        self.proj_prog_len = nn.Linear(self.d_hidden, self.pcbit)
        
    def forward(self, z_hidden, tau=1.0):
        B = z_hidden.size(0)
        h = z_hidden
        R_head = torch.sigmoid(self.proj_R(h)).view(B, -1, self.n_regs, self.n_val)
        prog_len = torch.sigmoid(self.proj_prog_len(h)).view(B, -1, self.pcbit)
        prog_threshold = torch.sigmoid(self.proj_prog_thres(h)).view(B,-1,1)
        
        logits = {
            "op": torch.sigmoid(self.proj_op(h)).view(B, -1, self.prog_max_length, self.n_op),
            "dst": torch.sigmoid(self.proj_dst(h)).view(B, -1, self.prog_max_length, self.n_dst),
            "src1": torch.sigmoid(self.proj_src1(h)).view(B, -1, self.prog_max_length, self.n_src1),
            "src2": torch.sigmoid(self.proj_src2(h)).view(B, -1, self.prog_max_length, self.n_src2),
            "imm" : torch.sigmoid(self.proj_imm(h)).view(B, -1, self.prog_max_length, self.n_imm),
        }
        
        return logits, R_head, prog_len, prog_threshold
        
            

if __name__ == '__main__':
    l = Augmentation_Layer(3,3,8,16,8,0.1)
    z_hidden = torch.zeros(1,3)
    l(z_hidden)
    
    # l = bits16_to_int(torch.tensor([[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]]))
        