import torch
import torch.nn as nn
import torch.nn.functional as F
from NEM.crossAttention import Prog2Transformer,InitialParse
from NEM.registers import Registers
from types import SimpleNamespace
import math


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
        self.register_buffer("ema_gate_mean", None)
        
    
    def forward(self, hidden_states, epoch):
        if epoch < self.warmup_epochs:
            gamma = 0.5 * (1 + math.cos(math.pi * (1 - epoch / self.warmup_epochs))) + self.args.eps
        else:
            gamma = 1.0
            
        current_gate = torch.sigmoid(self.proj(hidden_states))
        
        if self.args.gate_ema:
            batch_mean = current_gate.detach().mean(dim=0, keepdim=True)   
            if self.ema_gate_mean is None:
                with torch.no_grad():
                    self.ema_gate_mean = batch_mean.clone()
                gate = current_gate
                
            else:
                ema_expanded = self.ema_gate_mean.expand_as(current_gate)    
                gate = self.alpha * ema_expanded + (1 - self.alpha) * current_gate
       
                with torch.no_grad():
                    self.ema_gate_mean.mul_(self.ema_decay).add_(batch_mean, alpha=1 - self.ema_decay)
                
        else:
            gate = current_gate
            
        return gate * gamma, (gate * gamma)[0].item(), None
        
    def generate(self, hidden_states):
        return torch.sigmoid(self.proj(hidden_states))
      


class NeuralExecutionModuleWithSharedModule(nn.Module):
    def __init__(self,config,args,shareFeatureProj, shareInterpreter):
        super().__init__()
        self.config = config
        self.args = args
        self.InitialParse = InitialParse(config)
        self.FeatureProj = shareFeatureProj
        self.Interpreter = shareInterpreter
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
    
  