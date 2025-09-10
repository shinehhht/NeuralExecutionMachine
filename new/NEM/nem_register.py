import torch
import torch.nn as nn
import torch.nn.functional as F
from NEM.crossAttention import InitialParse, Prog2Transformer
from NEM.registers import Registers
from NEM.featurePproj import FeatureProjWithRegisters
from NEM.interpreter import InterpreterWithRegisters
from types import SimpleNamespace


config = SimpleNamespace(
   hidden_dim = 1024,
   num_heads = 8,
   dropout = 0.3,
   total_slots = 18,
   num_instructions = 4,
   num_op = 2,
   num_cycles = 2,
   instruction_types = 4,
   total_slots_registers=98,
   n_regs = 8
)


class NeuralExecutionModuleWithRegisters(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.InitialParse = InitialParse(config)
        self.FeatureProj = FeatureProjWithRegisters(config)
        self.Interpreter = InterpreterWithRegisters(config)
        self.Fuse2Main = Prog2Transformer(config)
       
        
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        
        registers = Registers(self.config, batch_size)
        
        value_initial, k_initial, gate, k_write, q_read, opcode, cond = self.InitialParse(hidden_states)

        registers.write(value_initial,k_initial)
        opcode_probs, cond_distribution, gate = self. FeatureProj(opcode, cond, gate)
        update_registers,program_details = self.Interpreter(opcode_probs, registers, k_write, q_read, cond_distribution, gate)
       
        fused_hidden_states = self.Fuse2Main(hidden_states, update_registers)

        return fused_hidden_states
                
                
        
                      
if __name__ == '__main__':
    hidden_states = torch.randn(2, 10, 1024, dtype=torch.float32)
    nem = NeuralExecutionModuleWithRegisters(config)
    nem(hidden_states)