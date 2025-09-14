import torch
import torch.nn as nn



class Instruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, args, p_op, state):
        pass
    

class Arithmetic(Instruction):
    def __init__(self, lookup_table, op_ids, config):
        super().__init__(config)
        self.lookupTable = lookup_table
        self.op_ids = op_ids
        self.register_write_gate = torch.ones(len(op_ids)) 
        
    def forward(self, args, p_op, state):
        """
        args: {"p_dst": (b,r),
                "p_s1": (b,r),
                "p_s2": (b,r)}
                
        """

        u = state.R.read(args["p_s1"]) # (b,n_vals)
        v = state.R.read(args["p_s2"])
        # print(f"u is {u[0]}")
        out = torch.einsum("bo,bi,bj,oijn->bn", p_op, u, v, self.lookupTable)
        # 
        w_prob = torch.einsum("bo,o->b",p_op,self.register_write_gate).unsqueeze(-1)
        # print(f"w_prob {w_prob}")
        print(f"arithmetic out is {out}")
        # state.R.write(args["p_dst"], out,w_prob)
        
        return out, w_prob
        # return state #out
    
class Compare(Instruction):
    def __init__(self, op_ids, config):
        super().__init__(config)
        self.op_ids = op_ids
        self.value_size = config.value_size
        self.TRI = torch.tril(torch.ones(self.value_size,self.value_size), -1)
        self.idx =  {
            "cmp_eq":0,
            "cmp_lt":1,
            "cmp_gt":2
        }
        self.register_write_gate = torch.ones(len(op_ids)) # hard code
        
    def forward(self, args, p_cmp, state):
        """
        p_cmp: (B,3)
        args: {"p_dst": (b,r),
                "p_s1": (b,r),
                "p_s2": (b,r)}
        """

        print(f"now doing compare")
        u = state.R.read(args["p_s1"]) # (b,n_vals)
        v = state.R.read(args["p_s2"])

        peq = p_cmp[:, self.idx["cmp_eq"]:self.idx["cmp_eq"]+1]  # (B,1)
        plt = p_cmp[:, self.idx["cmp_lt"]:self.idx["cmp_lt"]+1]
        pgt = p_cmp[:, self.idx["cmp_gt"]:self.idx["cmp_gt"]+1]
        
       #print(f"probability {peq} {plt} {pgt}")
        # eq u和v完全一致
        eq_prob = (u*v).sum(-1, keepdim=True)    # (B,1)
        
        lt_prob = torch.einsum("bw,kw->bk", u, self.TRI) * v # (B, n_vals) [P(u<0), ... P(u < n_vals-1)]
        lt_prob = lt_prob.sum(-1, keepdim=True)  # (B,1)
        
        gt_prob = torch.einsum("bw,kw->bk", v, self.TRI) * u 
        gt_prob = gt_prob.sum(-1, keepdim=True)  # (B,1)
        
        flag_prob = peq * eq_prob + plt * lt_prob + pgt * gt_prob
        flag = torch.zeros(len(flag_prob), 10)
        flag[:,1] = flag_prob.squeeze(1)
        print(f"flag is {flag}")
        # print(f"register before is {state.R.registers}")
        
        w_prob = torch.einsum("bo,o->b",p_cmp,self.register_write_gate).unsqueeze(-1) # (B,1)
        # state.R.write(args["p_dst"], flag, w_prob)
        # print(f"register now is {state.R.registers}")
        print(f"w_prob {w_prob}")
        return flag, w_prob
        return state

class Control(Instruction):
    def __init__(self, op_ids, config):
        super().__init__(config)
        self.config = config
        self.op_ids = op_ids
        self.idx =  {
            "jump":0,
            "halt":1,
        }
        self.register_write_gate = torch.zeros(len(op_ids))
        
    def forward(self, args, p_control, state):
        print("now doing control")
        p_jump = p_control[:, self.idx["jump"]:self.idx["jump"]+1]
        # print(f"p_jump is {p_jump}")
        p_halt = p_control[:, self.idx["halt"]:self.idx["halt"]+1]
            
        p_cond = args["p_cond"]
        print(f"condlist is {state.R.read(p_cond)}")
        cond = state.R.read(p_cond)[:, 1:2] 
        # print(f"cond is {cond}")
        state.h = state.h + (1 - state.h) * p_halt        
        pc_inc = torch.roll(state.pc, shifts=1, dims=1)
        
        print(f"p_jump {p_jump}")
        print(f"p_addr {args['p_addr']}")
        print(f"cond is {cond}")
        state.pc = (1 - p_jump) * pc_inc + p_jump * ((1 - cond) * pc_inc + cond * args["p_addr"])
        print(f"pc is {state.pc}")
        dummy_val = torch.zeros_like(state.R.read(args["p_dst"]))
        dummy_w   = torch.zeros(self.config.batch_size, 1)
        # print(f"dummy {dummy_w}")
        return dummy_val, dummy_w
        # return state
        