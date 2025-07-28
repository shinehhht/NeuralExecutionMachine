from library import make_program_demo,make_program_demo_jump,make_program_demo_compare
from interpreter import Interpreter
from computer import DifferentiableComputer, State
from config import config
import torch

n_regs = config.n_regs
n_vals = config.value_size
batch_size = config.batch_size
OP_SET = config.OP_SET

program = make_program_demo_compare(n_regs,n_vals)
interpreter = Interpreter(op_set= OP_SET, n_regs=n_regs, value_size=n_vals, prog=program,config=config)
computer = DifferentiableComputer(interpreter,10,0.99)

state = State(batch_size, n_regs, n_vals, program.size(0), address_num=0)
# print(state.R.registers.shape)

"""
jump_test:
with torch.no_grad():
    state.R.registers[0, 2] = 1.    # r0 = 2
    state.R.registers[1, 6] = 1.    # r1 = 6
    state.R.registers[2, 1] = 1.    # r2 = 1
"""
"""
# add_test
with torch.no_grad():
    state.R.registers[2, 2] = 1.    # r0 = 2
    state.R.registers[1, 6] = 1.    # r1 = 6

"""
# test compare
with torch.no_grad():
    
    state.R.registers[0, 2] = 1.    # r0 = 2
    state.R.registers[1, 6] = 1.    # r1 = 6
    state.R.registers[4, 1] = 1.    # r4 = 1
    state.R.registers[5, 1] = 1.    # r5 = 1
    
 
state = computer.run(state,10)

print(state.R.read(torch.tensor([[ 0.,0.,0.,1.,0.,0.]])).argmax(1).item())