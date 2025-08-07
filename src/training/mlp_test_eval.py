import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_model.mlp_test_model import MiniModel #RegulateRegisterModel
from isa.isa_parse import Parser
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = RegulateRegisterModel(8,16,16,0.3,64,64).to(device)
model = MiniModel(config).to(device)
model.load_state_dict(torch.load("mlp-test/mlp_model_fieldsample_tiny_group30_100epoch_actual.pth"))
model.eval()


input = torch.tensor([[9.,7.]],device=device)

total_bit = config.n_op + config.n_dst + config.n_src1 + config.n_src2 + config.n_imm
    
field_bits = {
    'op': config.n_op,
    'dst': config.n_dst,
    'src1': config.n_src1,
    'src2': config.n_src2,
    'imm': config.n_imm,
}
instruction = ["add r0 r1 r2 0"]
parser = Parser(config)
P = parser.risc2binary(instruction) # (1,1,len)
P = P.unsqueeze(-1)
P = ((P >> torch.arange(total_bit-1, -1, -1)) & 1).int().to(device)

logits = {}
offset = 0
for field, width in field_bits.items():
    logits[field] = P[..., offset : offset + width]
    offset += width

proglen = torch.zeros(1,1,config.pc_bit, dtype=torch.int32).to(device)  #(1,1,2)
proglen[..., -1] = 1

output,program,_ = model.generate(input)

print(f'answer is {output.item()}')

P = program['P'] # (1,1,max_length)
R = program['R'] # (1,1,n_regs)
prog_len = program['prog_len']

true_program = parser.binary2risc(P.squeeze(1))
true_registers = parser.binary2register(R.squeeze(1))
print(f"Program is \n{true_program}\n")
print(f"program length is \n{prog_len.item()}\n")
print(f"assign register is \n{true_registers}")
