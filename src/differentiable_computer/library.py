import torch
from config import config

"""
op | dst | src1 | src2 | cond_reg | jump_addr 
        
"""
OP_SET = config.OP_SET

def make_program_demo(n_regs, n_vals):
    n_ops = 8
    param_dim = n_ops + 4 * n_regs + 2
    P = torch.full((2, param_dim), -100.)
    
    P[0,OP_SET.index('Add')] = 100.
    P[0,n_ops + 0] = 100. # dst
    P[0,n_ops + n_regs + 1] = 100. # R1
    P[0,n_ops + 2*n_regs + 2] = 100. # R2
    
    P[1,OP_SET.index('halt')] = 100.
    
    return P

def make_program_demo_jump(n_regs, n_vals):
    """
    0: jump r2  2     
    1: add  r3 r1 r0
    2: del  r3 r1 r0
    3: halt
    """
    n_ops = len(config.OP_SET)
    param_dim = n_ops + 4 * n_regs + 4
    P = torch.full((4, param_dim), -100.)
    
    P[0,OP_SET.index('jump')] = 100.
    P[0,n_ops + 3*n_regs + 2] = 100.
    P[0,n_ops + 4*n_regs + 2] = 100.
    
    P[1,OP_SET.index('Add')] = 100.
    P[1,n_ops + 3] = 100. # dst
    P[1,n_ops + n_regs + 1] = 100. # src1
    P[1,n_ops + 2*n_regs + 0] = 100. # src2
    
    
    P[2,OP_SET.index('Sub')] = 100.
    P[2,n_ops + 3] = 100. # dst
    P[2,n_ops + n_regs + 1] = 100. # src1
    P[2,n_ops + 2*n_regs + 0] = 100. # src2
    
    P[3,OP_SET.index('halt')] = 100.
    
    return P

def make_program_demo_compare(n_regs,n_vals):
    """
    0: cmp_eq r2 r4 r5
    1: jump r2  3     
    2: add  r4 r1 r0
    3: del  r3 r1 r0
    4: halt
    
    """
    n_ops = len(config.OP_SET)
    param_dim = n_ops + 4 * n_regs + 5
    P = torch.full((5, param_dim), -100.)
    
    P[0,OP_SET.index('cmp_eq')] = 100.
    P[0,n_ops + 2] = 100.
    P[0,n_ops + n_regs + 4] = 100.
    P[0,n_ops + 2*n_regs + 5] = 100.
    
    
    P[1,OP_SET.index('jump')] = 100.
    P[1,n_ops + 3*n_regs + 2] = 100.
    P[1,n_ops + 4*n_regs + 3] = 100.
    
    P[2,OP_SET.index('Add')] = 100.
    P[2,n_ops + 4] = 100. # dst
    P[2,n_ops + n_regs + 1] = 100. # src1
    P[2,n_ops + 2*n_regs + 0] = 100. # src2
    
    
    P[3,OP_SET.index('Sub')] = 100.
    P[3,n_ops + 3] = 100. # dst
    P[3,n_ops + n_regs + 1] = 100. # src1
    P[3,n_ops + 2*n_regs + 0] = 100. # src2
    
    P[4,OP_SET.index('halt')] = 100.
    
    return P
    