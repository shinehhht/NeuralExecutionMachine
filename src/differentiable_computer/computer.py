import torch
import torch.nn as nn
from instructions import Instruction
from memory import Memory, Register

class State():
    def __init__(self, batch_size, n_regs, value_size, prog_len, address_num):
        self.R = Register(n_regs,value_size)
        self.Memory = Memory(address_num, value_size)
        self.pc = torch.zeros(batch_size, prog_len)
        self.pc[:,0] = 1.0
        self.h = torch.zeros(batch_size,1)
        
        
class DifferentiableComputer(nn.Module):
    def __init__(self, interpreter, max_steps=128, halt_threshold=0.99):
        super().__init__()
        self.interpreter = interpreter
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        
    def run(self,state,steps):
        steps = steps or self.max_steps
        
        for i in range(steps):
            if torch.all(state.h  > self.halt_threshold):
                break
            print(f"step {i+1}")
            state = self.step(state)

        return state
    
    def step(self, state):
        return self.interpreter(state)
                
        
