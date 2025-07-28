import torch
import torch.nn as nn
import torch.nn.functional as F
from instructions import Arithmetic, Compare, Control

def build_lookup(n):
    
    T = torch.zeros(3, n, n, n)
    for o, op in enumerate(["Add", "Sub", "Mul"]):
        for i in range(n):
            for j in range(n):
                if op == "Add":
                    k = (i + j) % n
                elif op == "Sub":
                    k = (i - j) % n
                elif op == "Mul":
                    k = (i * j) % n
                else:                      
                    continue
                T[o, i, j, k] = 1.
    return T

write_instruction_type_list = ["Arithmetic", "Compare"]

def build_instructions(config):
    op_id = {type:i for i, type in enumerate(config.OP_SET)}
    lookup_table = build_lookup(config.value_size)
    
    instructions = [
        Arithmetic(lookup_table, {k:op_id[k] for k in ["Add","Sub","Mul"]}, config),
        Compare({k:op_id[k] for k in ["cmp_eq","cmp_lt","cmp_gt"]}, config),
        Control({k:op_id[k] for k in ["jump","halt"]}, config)
    ]
    
    return instructions


class Interpreter(nn.Module):
    def __init__(self,op_set, n_regs, value_size, prog, config):
        """
        op | dst | src1 | src2 | cond_reg | jump_addr 
        
        """
        super().__init__()
        self.config = config
        self.n_regs = n_regs
        self.value_size = value_size
        self.op_set = op_set
        self.prog = prog
        # print(f"lookup {self.lookup_table[0][2][3]}")
        self.n_ops = len(op_set)
        self.ops_slice = slice(0, self.n_ops)
        self.dst_slice = slice(self.n_ops, self.n_ops + n_regs)
        self.src1_slice = slice(self.n_ops + n_regs, self.n_ops + 2*n_regs)
        self.src2_slice = slice(self.n_ops + 2*n_regs, self.n_ops + 3*n_regs)
        
        # add conditional jump
        self.cond_slice = slice(self.n_ops + 3*n_regs, self.n_ops + 4*n_regs)
        self.jump_addr_slice = slice(self.n_ops + 4*n_regs, self.n_ops + 4*n_regs + len(prog))
        self.op_id = {op: i for i, op in enumerate(op_set)}
        
    def forward(self, state):
        
        line_logits = torch.einsum("bp,pd->bd",state.pc, self.prog) # (b,param_dim)
        # print(line_logits)
        p_op   = F.softmax(line_logits[:, self.ops_slice],dim=-1)   # (b,A)
        print(f"p_op distribution {p_op}")
        p_dst  = F.softmax(line_logits[:, self.dst_slice],dim=-1)    # (b,r)
        p_s1   = F.softmax(line_logits[:, self.src1_slice],dim=-1)    # (b,r)
        p_s2   = F.softmax(line_logits[:, self.src2_slice],dim=-1)    # (b,r)
        p_cond = F.softmax(line_logits[:, self.cond_slice], dim=-1)   # (b, n_regs)
        print(f"p_cond {p_cond}")
        p_addr = F.softmax(line_logits[:, self.jump_addr_slice], dim=-1)   # (b, n_lines)
        print(f"p_addr {p_addr}")
        instructions = build_instructions(self.config)
        
        
        value_l = []
        pw_l = []
        
        for i, instruction in enumerate(instructions):
            print(f"For instruction {instruction}")
            cols = torch.tensor([self.op_id[k] for k in instruction.op_ids])
            print(cols)
            p_instruction = p_op[:,cols]
            print(f"distribution is {p_instruction}\n\n")
            args = {
                "p_dst": p_dst,
                "p_s1": p_s1,
                "p_s2": p_s2,
                "p_cond": p_cond,
                "p_addr": p_addr 
            }

            out,pw = instruction(args, p_instruction, state)
            print(f"{i}th {pw}")
            value_l.append(out)
            pw_l.append(pw)
            # state = instruction(args, p_instruction, state)
            # ans[instruction_type_list[i]] = out
            
        state.R.update_concurrent(p_dst, value_l, pw_l)
        
        print(f"now Register is {state.R.registers}")
        """
        p_jump = p_op[:, self.op_id["jump"]: self.op_id["jump"]+1]   # (b,1)   

        cond = state.R.read(p_cond)[:, 1:2] 
        
        u = state.R.read(p_s1) # (b,n_vals)
        v = state.R.read(p_s2)

        out = torch.einsum("bo,bi,bj,oijn->bn", p_op, u, v, self.lookup_table)
        # 
        # print(out)
        state.R.write(p_dst, out)

        p_halt = p_op[:, self.op_id["halt"] : self.op_id["halt"]+1]   # (b,1)
        state.h = state.h + (1 - state.h) * p_halt        
        pc_inc = torch.roll(state.pc, shifts=1, dims=1)
        
        #print(f"p_addr {p_addr}")
        #print(p_jump)  
        #print(f"pc_inc {pc_inc}")  
        state.pc = (1 - p_jump) * pc_inc + p_jump * ((1 - cond) * pc_inc + cond * p_addr)
        #print(f"pc distribution is {state.pc}")
        """
        
        return state    
        
        
        