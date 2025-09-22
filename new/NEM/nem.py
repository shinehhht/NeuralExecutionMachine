import torch
import torch.nn as nn
import torch.nn.functional as F
from NEM.crossAttention import Transformer2Prog, Prog2Transformer,InitialParse
from NEM.registers import Registers
from NEM.featurePproj import FeatureProj,FeatureProjWithRegisters
from NEM.interpreter import Interpreter,InterpreterWithRegisters
from types import SimpleNamespace
import math

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
                

class GateProj(nn.Module):
    def __init__(self,config,args):
        super().__init__()
        self.config = config
        self.args = args
        self.warmup_epochs = getattr(config, 'warmup_epochs', 30)
        self.proj = nn.Linear(config.hidden_dim,1)

        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        
        self.ema_decay = args.gate_decay
        self.alpha = args.gate_alpha
        self.register_buffer("ema_gate_portion", None)
        
    
    def forward(self, hidden_states, epoch):
        
        if epoch < self.warmup_epochs:
            gamma = 0.5 * (1 + math.cos(math.pi * (1 - epoch / self.warmup_epochs))) + self.args.eps
        else:
            gamma = 1.0
            
        current_gate = torch.sigmoid(self.proj(hidden_states))
        
        if self.args.gate_ema:
            if self.ema_gate_portion is None:
                gate = current_gate
                with torch.no_grad():
                    self.ema_gate_portion = current_gate.detach().clone()
            else:
                gate = self.alpha * self.ema_gate_portion.detach() + (1 - self.alpha) * current_gate
                with torch.no_grad():
                    self.ema_gate_portion.mul_(self.ema_decay)
                    self.ema_gate_portion.add_(current_gate.detach(), alpha=1-self.ema_decay)
        else:
            gate = current_gate
            
        if not self.args.constrain:
            return gate * gamma, (gate * gamma)[0].item(), None
        else:
            final_gate = self.schedule((gate * gamma),epoch)
            return final_gate,(gate * gamma)[0].item(), final_gate[0]
    
    def schedule(self,gate,epoch):
        
        T1 = self.args.T1
        T2 = self.args.T2
        U1 = self.args.U1
        L = self.args.L
        U2 = self.args.U2
        
        
        if epoch <= T1:
            low, high = 0.0, U1
        
        elif epoch < T2:
            alpha = (epoch - T1) / (T2 - T1)
            low  = 0.0   + alpha * ( L - 0.0)
            high = U2    + alpha * (1.0 - U2)
    
        else:
            low, high = 0.0, 1.0
        
        
        return low + (high - low) * gate
    def generate(self, hidden_states):
        return torch.sigmoid(self.proj(hidden_states))
             
class NeuralExecutionModuleWithRegisters(nn.Module):
    def __init__(self,config,args):
        super().__init__()
        self.config = config
        self.args = args
        self.InitialParse = InitialParse(config)
        self.FeatureProj = FeatureProjWithRegisters(config)
        self.Interpreter = InterpreterWithRegisters(config,args)
        self.Fuse2Main = Prog2Transformer(config)
        self.Gate = GateProj(config,args)

        self.ln_registers = nn.LayerNorm(self.config.hidden_dim)
        
    def forward(self, hidden_states,epoch):
        batch_size = hidden_states.shape[0]
        
        registers = Registers(self.config, batch_size)
        final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode, cond = self.InitialParse(hidden_states)

        registers.write(value_initial,k_initial)
        
        
        opcode_probs, cond_distribution, gate = self.FeatureProj(opcode, cond, gate)
        update_registers,program_details = self.Interpreter(opcode_probs, registers, k_write, q_read, cond_distribution, gate)

        fusing_gate, origin_gate, processed_gate = self.Gate(final_registers_gate,epoch)
        fusing_gate = fusing_gate.unsqueeze(-1)
        
        
        # gated_updated_registers = self.ln_registers(fusing_gate * update_registers)
        gated_updated_registers = fusing_gate * self.ln_registers(update_registers)
        fused_hidden_states = self.Fuse2Main(hidden_states, gated_updated_registers)

        return fused_hidden_states, origin_gate, processed_gate, program_details
    
    def generate(self, hidden_states):
        batch_size = hidden_states.shape[0]
        
        registers = Registers(self.config, batch_size)
        
        final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode, cond = self.InitialParse(hidden_states)

        registers.write(value_initial,k_initial)
        opcode_probs, cond_distribution, gate = self. FeatureProj(opcode, cond, gate)
        update_registers,program_details = self.Interpreter(opcode_probs, registers, k_write, q_read, cond_distribution, gate)
        
        update_registers = self.ln_registers(update_registers)
        fusing_gate = self.Gate.generate(final_registers_gate).unsqueeze(-1)
        
        gated_updated_registers = fusing_gate * update_registers
        fused_hidden_states = self.Fuse2Main(hidden_states, gated_updated_registers)

        return fused_hidden_states
    
                      
if __name__ == '__main__':
    hidden_states = torch.randn(2, 10, 1024, dtype=torch.float32)
    nem = NeuralExecutionModule(config)
    nem(hidden_states)