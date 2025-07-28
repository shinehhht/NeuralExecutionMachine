import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Convert, Programable

    
def gumbel_one_hot(logits, tau=1.0):
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)


class Augmentation_Layer(nn.Module):
    def __init__(self, d_model, d_hidden, n_regs, n_val, prog_max_length, dropout):
        """
        hidden_dim: (B,d_model) layer output by LLM
        
        """
        super().__init__()
        self.d_hidden = d_model # d_hidden or d_model
        self.n_regs = n_regs
        self.n_val = n_val
        self.prog_max_length = prog_max_length
        self.dropout = dropout
        
        """
        self.proj = nn.Sequential(
            nn.Linear(d_model, self.d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        """
        
        self.proj_prog_thres = nn.Linear(self.d_hidden,1)
        nn.init.constant_(self.proj_prog_thres.weight, 1e-3)  
        nn.init.constant_(self.proj_prog_thres.bias, -1.0)


        self.proj_R = nn.Linear(self.d_hidden, n_regs*n_val) # (B, L, n_regs*n_val)
        # self.proj_M = nn.Linear(self.d_hidden, n_mem*n_val)
        self.proj_op   = nn.Linear(self.d_hidden, prog_max_length * 16)  
        self.proj_dst  = nn.Linear(self.d_hidden, prog_max_length *  8)
        self.proj_src1 = nn.Linear(self.d_hidden, prog_max_length *  8)
        self.proj_src2 = nn.Linear(self.d_hidden, prog_max_length *  8)
        self.proj_imm  = nn.Linear(self.d_hidden, prog_max_length *  8)
        self.proj_prog_len = nn.Linear(self.d_hidden, prog_max_length)
        self.converter = Convert.apply
        self.programable = Programable.apply
        
        self.op_extract = torch.arange(16).view(16,1).float()
        
    def forward(self, z_hidden, tau=1.0):
        B = z_hidden.size(0)
        # h = self.proj(z_hidden)
        h = z_hidden
        R_head = torch.sigmoid(self.proj_R(h)).view(B, -1, self.n_regs, self.n_val)
        # M_head = torch.sigmoid(self.proj_M(h)).view(B, self.n_regs, self.n_val)
        prog_len = F.softmax(self.proj_prog_len(h).view(B, -1, self.prog_max_length), dim=-1)
        prog_threshold = torch.sigmoid(self.proj_prog_thres(h)).view(B,-1,1)
        
        logits = {
            "op": self.proj_op(h).view(B, -1, self.prog_max_length, 16),
            "dst": self.proj_dst(h).view(B, -1, self.prog_max_length, 8),
            "src1": self.proj_src1(h).view(B, -1, self.prog_max_length, 8),
            "src2": self.proj_src2(h).view(B, -1, self.prog_max_length, 8),
            "imm" : self.proj_imm(h).view(B, -1, self.prog_max_length, 8),
        }
        
        g_one_hot = lambda L: gumbel_one_hot(L, tau)   
        onehots   = {k: g_one_hot(v) for k, v in logits.items()}
        print(f"onehots is {onehots}")
        # print(f"op hot is {onehots['op']}")
        
        #op = onehots['op'] @ self.op_extract
        #print(f"op is {op}, shape is {op.shape}")
        idx       = {k: oh.argmax(-1) for k, oh in onehots.items()} 
        # idx       = {k: self.ste_argmax(oh) for k, oh in onehots.items()} 
        # print(f"idx is {idx}")

        program_assemble = ((idx["op"]  << 12) | (idx["dst"] <<  9) | (idx["src1"]<<  6) | (idx["src2"]<<  3) | idx["imm"]) # (B, L, max_length)
        # program_assemble = self.programable(onehots)
        # print(f"program shape is {program_assemble.shape}")
        # R = bits16_to_int(R_head)
        R = self.converter(R_head)
        
        print(f"R shape is {R.shape}")
        M = torch.zeros(B, z_hidden.size(1), 2**self.n_val, self.n_val)
        # print(R.shape) # (B,n_reg)
        #print(R.requires_grad)
        # print(M.shape)
        # print(f"len {prog_len.shape}")
        return prog_threshold, program_assemble, R, M, prog_len
        
            

if __name__ == '__main__':
    l = Augmentation_Layer(3,3,8,16,8,0.1)
    z_hidden = torch.zeros(1,3)
    l(z_hidden)
    
    # l = bits16_to_int(torch.tensor([[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]]))
        