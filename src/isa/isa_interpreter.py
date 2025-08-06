import torch
import torch.nn as nn
import torch.nn.functional as F
from isa.isa_parse import Parser
from isa.isa_instructions import op_add, op_sub, op_mul, op_cmp_eq, op_cmp_gt, op_cmp_lt, op_jmp, op_mov
"""
op | dst | src1 | src2 | imm

op: 3 bit
dst: 2 bit
src1: 2 bit
src2: 2 bit
imm: 2 bit

11 bit
"""

def decode_instructions(instructions, field_order):
    total_bits = sum([width for width in field_order.values()])
    bit_offest = 0
    fields = {}
    instr = instructions.squeeze(-1) 
    
    for name,width in field_order.items():
        shift = total_bits - bit_offest - width
        mask = (1 << width) - 1
        fields[name] = ((instr >> shift) & mask).unsqueeze(-1)
        bit_offest += width
    
    return fields
    
class Interpreter:
    def __init__(self, registers, memory, program, program_len,config):
        self.device = registers.device
        self.B = registers.size(0)
        self.registers = registers.to(self.device)    # (B, L,n_regs)
        self.memory = memory.to(self.device)         # (B, L,n_mem)
        self.program = program.to(self.device)       # (B, L,prog_len)
        # for general interpreter: self.program_len = torch.argmax(program_len,dim=-1).to(self.device)   # (B, L)
        self.program_len = program_len.to(self.device)
       
        self.pc = torch.zeros((self.B,registers.size(1)), dtype=int, device=self.device)  
        
        
        self.n_op = config.n_op
        self.n_dst = config.n_dst
        self.n_src1 = config.n_src1
        self.n_src2 = config.n_src2
        self.n_imm = config.n_imm
        
        self.field_order = {
            'op': self.n_op,
            'dst': self.n_dst,
            'src1': self.n_src1,
            'src2': self.n_src2,
            'imm': self.n_imm,
        }
        
        self.op_table = {
            0: op_add,
            1: op_sub,
            2: op_mul,
            3: op_cmp_eq,
            4: op_cmp_lt,
            5: op_cmp_gt,
            6: op_jmp,
            7: op_mov,
        }
        
    def run(self):
        i = 1
        while torch.any(self.pc < self.program_len): 
            self.step()
            i += 1
        return self.registers, self.memory
            
    def step(self):
        active_mask = (self.pc < self.program_len) # (B,L)
        current_instruction = torch.gather(self.program, dim=2, index=self.pc.unsqueeze(-1)) # (B,L,1)
        
        fields = decode_instructions(current_instruction, self.field_order)
        
        op = fields['op']
        dst  = fields['dst']
        src1 = fields['src1']
        src2 = fields['src2']
        imm  = fields['imm']
        
        r1 = self.registers.gather(2, src1) # (B,L,1)
        r2 = self.registers.gather(2, src2)
        
        result = torch.zeros_like(r1) 
        
        for op_code, instruction in self.op_table.items():
            # print(f"instruction is {instruction}")
            mask = (op == op_code) # (B,L,1)
            # print(f"mask is {mask}, mask shape is {mask.shape}")
            val,pc_delta = instruction(r1,r2,imm,active_mask)
            # print(f"val is {val}")
            result += mask * val
        active_mask = active_mask.unsqueeze(-1)
        #print(f"active mask shape is {active_mask.shape}")
        #print(f"result shape is {result.shape}")
        #print(f"registers shape is {self.registers.gather(2,dst).shape}")
        result = active_mask * result + (~active_mask) * self.registers.gather(2,dst)
        self.registers = self.registers.scatter(2, dst, result)
        
        self.pc += pc_delta
        

class MultiInterpreter:
    def __init__(self, registers, memory, program, program_len,config):
        self.device = registers.device
        self.B = registers.size(0)
        self.registers = registers.to(self.device)    # (B, L, group, n_regs)
        self.memory = memory.to(self.device)         # (B, L, group, n_mem)
        self.program = program.to(self.device)       # (B, L, group, prog_len)
        # for general interpreter: self.program_len = torch.argmax(program_len,dim=-1).to(self.device)   # (B, L)
        self.program_len = program_len.to(self.device) # (B,L,group)
       
        self.pc = torch.zeros((self.B,registers.size(1),registers.size(2)), dtype=int, device=self.device)   #(B,L,group)
        # print(f"pc shape {self.pc.shape}")
        
        self.n_op = config.n_op
        self.n_dst = config.n_dst
        self.n_src1 = config.n_src1
        self.n_src2 = config.n_src2
        self.n_imm = config.n_imm
        
        self.field_order = {
            'op': self.n_op,
            'dst': self.n_dst,
            'src1': self.n_src1,
            'src2': self.n_src2,
            'imm': self.n_imm,
        }
         
        # self.parser = Parser()
        self.op_table = {
            0: op_add,
            1: op_sub,
            2: op_mul,
            3: op_cmp_eq,
            4: op_cmp_lt,
            5: op_cmp_gt,
            6: op_jmp,
            7: op_mov,
        }
        
    def run(self):
        i = 1
        while torch.any(self.pc < self.program_len): 
            self.step()
            i += 1
        return self.registers, self.memory
            
    def step(self):
        # print("begin next step")
        active_mask = (self.pc < self.program_len) # (B,L,group)
        current_instruction = torch.gather(self.program, dim=3, index=self.pc.unsqueeze(-1)) # (B,L,group,1)
        
        fields = decode_instructions(current_instruction, self.field_order)
        
        op = fields['op']
        dst  = fields['dst']
        src1 = fields['src1']
        src2 = fields['src2']
        imm  = fields['imm']
        
        r1 = self.registers.gather(3, src1) # (B,L,G,1)
        r2 = self.registers.gather(3, src2)
        
        result = torch.zeros_like(r1) 
        
        for op_code, instruction in self.op_table.items():
            # print(f"instruction is {instruction}")
            mask = (op == op_code) # (B,L,G,1)
            # print(f"mask is {mask}, mask shape is {mask.shape}")
            val,pc_delta = instruction(r1,r2,imm,active_mask)
            # print(f"val is {val}")
            result += mask * val
        active_mask = active_mask.unsqueeze(-1)
        #print(f"active mask shape is {active_mask.shape}")
        #print(f"result shape is {result.shape}")
        #print(f"registers shape is {self.registers.gather(2,dst).shape}")
        result = active_mask * result + (~active_mask) * self.registers.gather(3,dst)
        self.registers = self.registers.scatter(3, dst, result)
        
        self.pc += pc_delta
        
        
        
        
if __name__ == '__main__':
    """
    p1:
    add r1 r2 r3 0     r2 = 3, r3 = 4
    sub r0 r1 r2 1
    
    p2:
    cmp_eq r7 r2 r3   r2=r3=3
    jmp r7 2
    add r0 r2 r3 0
    sub r1 r2 r3 1
    """
    parser = Parser()
    p1 = ["add r1 r2 r3 0", "sub r0 r1 r2 1", "add r1 r2 r3 0", "sub r1 r2 r3 1"]
    p2 = ["cmp_eq r7 r2 r3 0", "jmp r7 r3 r4 2","add r0 r2 r3 0", "sub r1 r2 r3 1"]
    
    p1b = parser.risc2binary(p1).unsqueeze(2)
    p2b = parser.risc2binary(p2).unsqueeze(2)
    registers1 = parser.assignRegister(2,3,3,4).unsqueeze(2)
    registers2 = parser.assignRegister(2,3,3,3).unsqueeze(2)
    
    program = torch.cat((p1b,p2b),dim=2)
    registers = torch.cat((registers1,registers2),dim=2)
    mem = torch.full_like(registers,1.0)
    prog_len = torch.zeros((1,1,2))
    prog_len[0,0,0] = 2
    prog_len[0,0,1] = 4
    print(f"program shape is {program.shape}")
    print(f"register shape is {registers.shape}")
    interpreter = MultiInterpreter(registers,mem,program,prog_len)
    interpreter.run()
    print(interpreter.registers.shape)