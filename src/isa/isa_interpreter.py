import torch
import torch.nn as nn
import torch.nn.functional as F
from isa.isa_parse import Parser
from isa.isa_instructions import op_add, op_sub, op_mul, op_cmp_eq, op_cmp_gt, op_cmp_lt, op_jmp, op_mov
"""
op | dst | src1 | src2 | imm

op: 4 bit
dst: 3 bit
src1: 3 bit
src2: 3 bit
imm: 3 bit

"""

def bits16_to_int(b):     
    hard = (b >= 0.5).int()
    b_st = b + (hard - b).detach()
    # print(b_st)
    weights = 2 ** torch.arange(15,-1,-1, device=b.device)
    unsigned_val = (b_st * weights).sum(-1)
    signed_val = torch.where(unsigned_val >= 2**15, unsigned_val - 2**16, unsigned_val)
    return signed_val


class Interpreter:
    def __init__(self, registers, memory, program, program_len):
        self.device = registers.device
        self.B = registers.size(0)
        self.registers = registers.to(self.device)    # (B, L,n_regs)
        self.memory = memory.to(self.device)         # (B, L,n_mem)
        self.program = program.to(self.device)       # (B, L,prog_len)
        self.program_len = torch.argmax(program_len,dim=-1).to(self.device)   # (B, L)
        # print(f"program length is {self.program_len.shape}")
        self.pc = torch.zeros((self.B,registers.size(1)), dtype=int, device=self.device)  
        # print(f"pc shape is {self.pc.shape}")        
        # self.parser = Parser()
        self.op_table = {
            0: op_add,
            1: op_sub,
            2: op_mul,
            3: op_cmp_eq,
            4: op_cmp_lt,
            5: op_cmp_gt,
            6: op_jmp,
            12: op_mov,
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
        
        op = (current_instruction >> 12) & 0xF  # (B,L,1)
        dst = (current_instruction >> 9) & 0x7
        src1 = (current_instruction >> 6) & 0x7
        src2 = (current_instruction >> 3) & 0x7
        imm = current_instruction & 0x7

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
        
        """
        current_instructions = [program[pc] for program, pc in zip(self.program, self.pc)]
        for idx, instruction in enumerate(current_instructions):
            with open("1.txt","a") as f:
                f.write(f"current instruction: {self.parser.parse_single(instruction)}\n")
            
            if self.pc[idx] > self.program_len[idx]:
                print("hhhhh")
                break
            
            op = (instruction >> 12) & 0xF
            dst = (instruction >> 9) & 0x7
            src1 = (instruction >> 6) & 0x7
            src2 = (instruction >> 3) & 0x7
            imm = instruction & 0x7

            
            if op == 0:  
                self.add(idx, dst, src1, src2)
            elif op == 1:  # sub
                self.sub(idx, dst, src1, src2)
            elif op == 2:  # mul
                self.mul(idx, dst, src1, src2)
            elif op == 3:  # cmp_eq
                self.cmp_eq(idx, dst, src1, src2)
            elif op == 4:  # cmp_lt
                self.cmp_lt(idx, dst, src1, src2)
            elif op == 5:  # cmp_lt
                self.cmp_gt(idx, dst, src1, src2)
            elif op == 6:  # jmp
                self.jmp(idx, imm)
            elif op == 7:  # halt
                self.halt(idx)
            elif op == 8:  # load
                self.load(idx, dst, src1, imm)
            elif op == 9:  # store
                self.store(idx, dst, src1, imm)
            elif op == 10:
                self.pc[idx] = self.program_len[idx]+2
                continue
            elif op == 11:
                self.pc[idx] = self.program_len[idx]+2
                continue
            elif op == 12:
                self.mv(idx, dst, src1)
            elif op == 13:
                pass
                #self.and_inst(idx,dst,src1,src2)
            elif op == 14:
                pass
                #self.or_inst(idx,dst,src1,src2)
            elif op == 15:
                pass
                #self.not_inst(idx,dst,src1)
                
            self.pc[idx] += 1
            with open("1.txt","a")as f:
                f.write(f"after this operation, pc is {self.pc[idx]}\n")
    
    def add(self, id, dst, src1, src2):
        self.registers[id, dst] = (self.registers[id, src1] + self.registers[id, src2]) % (2**15)

    def sub(self, id, dst, src1, src2):
        self.registers[id, dst] = self.registers[id, src1] - self.registers[id, src2]

    def mul(self, id, dst, src1, src2):
        self.registers[id, dst] = (self.registers[id, src1] * self.registers[id, src2]) % (2**15)

    def cmp_eq(self, id, dst, src1, src2):
        if self.registers[id, src1] == self.registers[id, src2]:
            self.registers[id, dst] = 1
        else:
            self.registers[id, dst] = 0

    def cmp_lt(self, id, dst, src1, src2):
        if self.registers[id, src1] < self.registers[id, src2]:
            self.registers[id, dst] = 1
        else:
            self.registers[id, dst] = 0
            
    def cmp_gt(self, id, dst, src1, src2):
        if self.registers[id, src1] > self.registers[id, src2]:
            self.registers[id, dst] = 1
        else:
            self.registers[id, dst] = 0
    def jmp(self, id, imm):
        if imm == 0 or 1:
            pass
        else:
            new = self.pc[id] + imm
            with open("1.txt","a")as f:
                f.write(f"imm is {imm}")
                f.write(f"new address is {new}")
            if 0<=new<=self.program_len[id]:
                self.pc[id] += imm-1
            else:
                print("non valid")

    def halt(self,id):
        #print("Execution halted.")
        self.pc[id] = self.program_len[id]
        pass
    
    def load(self, id, dst, src1, imm): 
        # load dst imm(src1)
        
        print(f"dst is {dst}")
        print(f"src1 is {src1}")
        print(f"imm is {imm}")
        print(f"src1 register is {self.registers[id, src1]}")
        print(f"total register {self.registers}")
        address = int(self.registers[id, src1] + imm)
        print(f"address is {address}")
        print(f"memory content is {self.memory[id, address]}")
        if 0<=address<65536:
            self.registers[id, dst] = bits16_to_int(self.memory[id, address])
        else:
            print("non valid")

    def store(self, id, dst, src1, imm):
        address = int(self.registers[id, src1] + imm)
        if 0<=address<65536:
            self.memory[id, address] = self.registers[id, dst]
        else:
            print("non valid")
    
    def mv(self, id, dst, src1):
        value = int(self.registers[id, src1])
        self.registers[id,dst] = value
        
    def and_inst(self, id, dst, src1, src2):
        self.registers[id, dst] = self.registers[id, src1] & self.registers[id, src2]
        
    def or_inst(self, id, dst, src1, src2):
        self.registers[id, dst] = self.registers[id, src1] | self.registers[id, src2]
        
    def not_inst(self, id, dst, src1):
            self.registers[id, dst] = ~self.registers[id, src1]
    """

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
    
    p1b = parser.risc2binary(p1)
    p2b = parser.risc2binary(p2)
    registers1 = parser.assignRegister(2,3,3,4)
    registers2 = parser.assignRegister(2,3,3,3)
    
    program = torch.cat((p1b,p2b),dim=1)
    registers = torch.cat((registers1,registers2),dim=1)
    prog_len = torch.zeros((1,2,16))
    prog_len[0,0,2] = 1.
    prog_len[0,1,4] = 1.
    print(f"program shape is {program.shape}")
    print(f"register shape is {registers.shape}")
    interpreter = Interpreter(registers,0,program,prog_len)
    interpreter.run()
    print(interpreter.registers)