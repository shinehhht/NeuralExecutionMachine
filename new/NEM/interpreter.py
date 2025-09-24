import torch
import torch.nn as nn
import torch.nn.functional as F
from NEM.instructions import Add, Sub

class Interpreter(nn.Module):
    def __init__(self, config,policy=None):
        super().__init__()
        self.config = config

        self.op1_proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1))
        self.op2_proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1))
        self.result_head = nn.Sequential(nn.Linear(1, config.hidden_dim))
        self.final_policy = policy

        self._ops = [
            lambda x, y: (x + y) % (2**15),
            lambda x, y: x - y,
            lambda x, y: (x * y) % (2**15),
            lambda x, y: x / (y.abs() + 1e-6),
        ]

    @staticmethod
    def _renorm_probs(p, dim = -1, eps= 1e-8):
        s = p.sum(dim=dim, keepdim=True)
        return p / (s + eps)

    def _apply_mask(self, op_feat_line, mask_line):
        gate = torch.sigmoid(mask_line)         # (b,2,d)
        masked = op_feat_line * gate
        return masked, gate


    def forward(self, opcode_probs, mask_distributions, cond_distributions, op):
        """
        opcode_prob (b,num_instructions,instruction_types)
        mask_distributions (b,2*num_instructions,d)
        cond_distributions (b,num_instructions,x)
        op (b,2,d)

        """
        b,lines,categories = opcode_probs.shape
        record_lines = []
        results_per_line = []

        for line in range(lines):
            prob = self._renorm_probs(opcode_probs[:, line, :], dim=-1) #(b,n)
            mask_s = mask_distributions[:,line:line+2,:]
            op_s, gate_s = self._apply_mask(op, mask_s)

            x = self.op1_proj(op_s[:, 0, :])  # (b,1)
            y = self.op2_proj(op_s[:, 1, :])  # (b,1)

            outs = []
            for i in range(categories):
                zi = self._ops[i](x, y)
                outs.append(zi)
            outs = torch.stack(outs, dim=1) #(b,n,1)

            mix = torch.einsum('bi, bid->bd', prob, outs) #(b,1)
            result_s = self.result_head(mix)
            results_per_line.append(result_s)


            record_lines.append({
                "opcode_probs_line": prob,   # (B,n)
                "mask_gate_line": gate_s,
                "arith_x": x, "arith_y": y,
                "outs_arith": outs,
            })

        results_per_line = torch.stack(results_per_line, dim=1)  # (b, num_instructions, d)

        if self.final_policy == "sum":
            final_result = results_per_line.sum(dim=1)           # (B,D)
        elif self.final_policy == "mean":
            final_result = results_per_line.mean(dim=1)          # (B,D)
        else:
            final_result = results_per_line[:, -1, :]            # (B,D)

        procedure = {
            "per_line": record_lines,
            "results_per_line": results_per_line,
        }
        return final_result, procedure

class InterpreterWithRegisters(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.config = config

        self.op1_proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1))
        self.op2_proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1))
        self.result_head = nn.Sequential(nn.Linear(1, config.hidden_dim))

        max_add_bit = args.output_bits + 1
        max_mul_bit = 2 * args.input_bits + 1
        
        self._ops = [
            lambda x, y: (x + y),
            lambda x, y: x - y,
            lambda x, y: (x * y),
            lambda x, y: x / (y.abs() + 1e-6),
        ]

        self.ln = nn.LayerNorm(self.config.hidden_dim)
        
    def forward(self, opcode_probs, registers, k_write, q_read, cond_distribution, gate):
        """
        opcode_prob (b,num_instructions,instruction_types)
        registers (b,n_reg,d)
        k_write (b,1,d)
        q_read (b,2,d)
        cond_distributions (b,num_instructions,x)
        gate (b,1,1)

        """
        b,lines,categories = opcode_probs.shape
        
        record_lines = []
        for line in range(lines):
            prob = opcode_probs[:, line, :] #(b,n)
            op_s = registers.read(q_read[:,line,:]) # (b,2,d)
            x = self.op1_proj(op_s[:, 0, :])  # (b,1)
            y = self.op2_proj(op_s[:, 1, :])  # (b,1)

            outs = []

            for i in range(categories):
                zi = self._ops[i](x, y)
                outs.append(zi)
            outs = torch.stack(outs, dim=1) #(b,n,1)
            mix = torch.einsum('bi, bid->bd', prob, outs) #(b,1)
            value = self.ln(self.result_head(mix).unsqueeze(1))

            registers.write(value,k_write[:,line:line+1,:],gate[:,line:line+1,:])

            
            record_lines.append({
                "opcode_probs_line": prob[0].tolist(),   
                "arith_x": x[0].item(), "arith_y": y[0].item(),
                "outs_arith": outs[0].tolist(),
                "actual_return_value": mix[0].item()
            })


        procedure = {
            "per_line": record_lines,
        }
        return registers.registers, procedure
    
    
class InterpreterWithRegistersAndKbit(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.config = config
        self.k = config.k_bits
        self.l = args.input_l_bits
        
        numerical_feature_space = self.l*self.k

        self.op1_proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, numerical_feature_space))
        self.op2_proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, numerical_feature_space))
        self.result_head = nn.Sequential(nn.Linear(numerical_feature_space, config.hidden_dim))
        
        self._ops = [
            Add(config, args),
            Sub(config, args)
        ]

        self.ln = nn.LayerNorm(self.config.hidden_dim)
        
    def forward(self, opcode_probs, registers, k_write, q_read, cond_distribution, gate):
        """
        opcode_prob (b,num_instructions,instruction_types)
        registers (b,n_reg,d)
        k_write (b,1,d)
        q_read (b,2,d)
        cond_distributions (b,num_instructions,x)
        gate (b,1,1)

        """
        b,lines,categories = opcode_probs.shape
        
        record_lines = []
        for line in range(lines):
            prob = opcode_probs[:, line, :] #(b,n)
            op_s = registers.read(q_read[:,line,:]) # (b,2,d)
            x = F.softmax(self.op1_proj(op_s[:, 0, :]).reshape(b,self.l,self.k),dim=-1)  # (b,l,k)
            y = F.softmax(self.op2_proj(op_s[:, 1, :]).reshape(b,self.l,self.k),dim=-1)  # (b,l,k)

            outs = []

            for i in range(categories):
                zi = self._ops[i](x, y) # (b,l,k)
                outs.append(zi)
            outs = torch.stack(outs, dim=1) #(b,n,l,k)
            mix = torch.einsum('bn, bnlk->blk', prob, outs) #(b,l,k)
            #print(f'mix shape is {mix.shape}')
            mix1 = mix.reshape(b,-1)
            value = self.ln(self.result_head(mix1).unsqueeze(1))

            registers.write(value,k_write[:,line:line+1,:],gate[:,line:line+1,:])

            
            record_lines.append({
                "opcode_probs_line": prob[0].tolist(),   
                "arith_x": x[0].tolist(), "arith_y": y[0].tolist(),
                "actual_return_value": mix[0].tolist()
            })


        procedure = {
            "per_line": record_lines,
        }
        return registers.registers, procedure