import torch

def op_add(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return (r1 + r2)%(2**15), pc_delta

def op_sub(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return r1 - r2, pc_delta

def op_mul(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return (r1 * r2)%(2**15), pc_delta

def op_mov(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return r1, pc_delta

def op_cmp_lt(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return r1 < r2, pc_delta

def op_cmp_eq(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return r1 == r2,pc_delta

def op_cmp_gt(r1, r2, imm, mask):
    pc_delta = mask * torch.ones_like(r1,dtype=torch.long).squeeze(-1)
    return r1 > r2, pc_delta

def op_jmp(r1, r2, imm, mask):
    cond = ((r1 - 1.) > -1e3) & ((r1 - 1.) < 1e3)
    pc_delta = torch.where(cond, imm, torch.ones_like(imm))  
   
    pc_delta = mask * pc_delta.squeeze(-1) # imm为0，陷入死循环
    
    return torch.zeros_like(r1), pc_delta
    