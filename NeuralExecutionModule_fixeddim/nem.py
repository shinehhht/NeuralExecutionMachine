import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from NeuralExecutionModule_fixeddim.crossAttention import Transformer2Prog, Prog2Transformer,InitialParse
from NeuralExecutionModule_fixeddim.crossAttentionGrouped import Prog2Transformer, InitialParseGrouped
from NeuralExecutionModule_fixeddim.registers import Registers
from NeuralExecutionModule_fixeddim.featurePproj import FeatureProj,FeatureProjWithRegisters
from NeuralExecutionModule_fixeddim.interpreter import Interpreter,InterpreterWithRegisters, InterpreterWithRegistersAndKbit
from types import SimpleNamespace
import math


class GateProj(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.warmup_steps = getattr(config, 'warmup_steps', 60)
        self.temperature = getattr(config, 'gate_temperature', 2.0)
        self.logit_clip = getattr(config, 'gate_logit_clip', 8.0)
        
        self.proj = nn.Linear(config.nem_dim,1)
        
        self.ema_decay = config.gate_decay
        self.alpha = config.gate_alpha
        self.register_buffer("ema_gate_mean", torch.zeros(1, 1))
        
    
    def forward(self, hidden_states, steps):
        """
        gamma_min = 0.1
        if steps <= self.warmup_steps:
            gamma = gamma_min + (1-gamma_min) * 0.5 * (1 + math.cos(math.pi * (1 - steps / self.warmup_steps)))
        else:
            gamma = 1.0
        """
        
        gamma = 1.0
        
        #current_gate = torch.sigmoid(self.proj(hidden_states))
        
        
        logits = self.proj(hidden_states)
        logits = logits.clamp(-self.logit_clip, self.logit_clip)
        current_gate = torch.sigmoid(logits/ self.temperature)
        scaled_gate = current_gate * gamma
        
        # batch_mean = current_gate.detach().mean(dim=0, keepdim=True)   
        batch_mean = scaled_gate.detach().mean(dim=0, keepdim=True)
        if torch.all(self.ema_gate_mean == 0):
            with torch.no_grad():
                self.ema_gate_mean = batch_mean.clone()
            gate = scaled_gate
            
        else:
            if steps < self.warmup_steps:
                effective_decay = self.ema_decay * 0.75
            else:
                effective_decay = self.ema_decay
                
            ema_expanded = self.ema_gate_mean.expand_as(current_gate)    
            gate = self.alpha * ema_expanded + (1 - self.alpha) * scaled_gate
    
            with torch.no_grad():
                self.ema_gate_mean.mul_(effective_decay).add_(batch_mean, alpha=1 - effective_decay)
                

        return gate
    
    """
    def forward(self, hidden_states, steps):
        gamma_min = 0.1
        if steps < self.warmup_steps:
            gamma = gamma_min + (1-gamma_min) * 0.5 * (1 + math.cos(math.pi * (1 - (steps-1) / self.warmup_steps)))
        else:
            gamma = 1.0
        
        logits = self.proj(hidden_states)
        logits = logits.clamp(-self.logit_clip, self.logit_clip)
        current_gate = torch.sigmoid(logits/ self.temperature)
        
        batch_mean = current_gate.detach().mean(dim=0, keepdim=True)   
        if torch.all(self.ema_gate_mean == 0):
            with torch.no_grad():
                self.ema_gate_mean = batch_mean.clone()
            gate = current_gate
            
        else:
            ema_expanded = self.ema_gate_mean.expand_as(current_gate)    
            gate = self.alpha * ema_expanded + (1 - self.alpha) * current_gate
    
            with torch.no_grad():
                self.ema_gate_mean.mul_(self.ema_decay).add_(batch_mean, alpha=1 - self.ema_decay)
                

        return gate * gamma
        """
    
class NeuralExecutionModuleWithRegistersAndKbit(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.InitialParse = InitialParseGrouped(config)
        #self.InitialParse = InitialParseGrouped(config)
        self.FeatureProj = FeatureProjWithRegisters(config)
        self.Interpreter = InterpreterWithRegistersAndKbit(config)
        self.Fuse2Main = Prog2Transformer(config)
        self.Gate = GateProj(config)

        self.ln_registers = nn.LayerNorm(self.config.nem_dim)
        
    def forward(self, hidden_states, steps):
        batch_size = hidden_states.shape[0]
        
        registers = Registers(self.config, batch_size)
        # final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode, cond = self.InitialParse(hidden_states)
        final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode = self.InitialParse(hidden_states)
        registers.write(value_initial,k_initial)
        
        
        opcode_probs,gate = self.FeatureProj(opcode, gate)
        update_registers,program_details = self.Interpreter(opcode_probs, registers, k_write, q_read, gate)

        fusing_gate = self.Gate(final_registers_gate, steps)
        fusing_gate = fusing_gate.unsqueeze(-1)
        gate_scalar = fusing_gate.mean()
        
        # gated_updated_registers = self.ln_registers(fusing_gate * update_registers)
        gated_updated_registers = fusing_gate * self.ln_registers(update_registers)
        fused_hidden_states = self.Fuse2Main(hidden_states, gated_updated_registers)

        return fused_hidden_states, gate_scalar, program_details
    
    def generate(self, hidden_states):
        batch_size = hidden_states.shape[0]
        
        registers = Registers(self.config, batch_size)
        
        final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode = self.InitialParse(hidden_states)

        registers.write(value_initial,k_initial)
        opcode_probs, gate = self.FeatureProj(opcode, gate)
        update_registers,program_details = self.Interpreter(opcode_probs, registers, k_write, q_read, gate)
        
        update_registers = self.ln_registers(update_registers)
        fusing_gate = self.Gate.generate(final_registers_gate).unsqueeze(-1)
        
        gated_updated_registers = fusing_gate * update_registers
        fused_hidden_states = self.Fuse2Main(hidden_states, gated_updated_registers)

        return fused_hidden_states, program_details
    
                      
if __name__ == '__main__':
    hidden_states = torch.randn(2, 10, 1024, dtype=torch.float32)
    nem = NeuralExecutionModule(config)
    nem(hidden_states)