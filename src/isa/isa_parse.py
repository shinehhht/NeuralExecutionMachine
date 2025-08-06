import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config
"""
op | dst | src1 | src2 | imm

op: 3 bit
dst: 2 bit
src1: 2 bit
src2: 2 bit
imm: 2 bit

"""
def decimal2binary(n):
    return format(n, '016b')

def decode_instruction(instruction, field_order):
    total_bits = sum([width for width in field_order.values()])
    bit_offest = 0
    fields = {}
    
    for name,width in field_order.items():
        shift = total_bits - bit_offest - width
        mask = (1 << width) - 1
        fields[name] = ((instruction >> shift) & mask)
        bit_offest += width
    
    return fields

class Parser():
    def __init__(self, config):
        super().__init__()
        
        self.n_regs = config.n_regs
        self.n_op = config.n_op
        self.n_dst = config.n_dst
        self.n_src1 = config.n_src1
        self.n_src2 = config.n_src2
        self.n_imm = config.n_imm
        self.n_val = config.n_val
        
        self.op_dict = {
            0:'add',
            1:'sub',
            2:'mul',
            3:'cmp_eq',
            4:'cmp_lt',
            5:'cmp_gt',
            6:'jmp',
            7:'mv'
        }
        
        self.op_dict2 = {
            "add": 0,
            "sub": 1,
            "mul": 2,
            "cmp_eq": 3,
            "cmp_lt": 4,
            "cmp_gt": 5,
            "jmp": 6,
            "mv": 7,
        }
        
        self.field_order = {
            'op': self.n_op,
            'dst': self.n_dst,
            'src1': self.n_src1,
            'src2': self.n_src2,
            'imm': self.n_imm,
        }
        
        self.register_dict = {i: f'r{i}' for i in range(self.n_regs)}
        
        
    def parse(self, P):
        """
        P (B,1,prog_len)
        """ 
        P = P.squeeze(1) # (B, prog_len)
        total_ans = []
        for p in P:
            output = []
            for line in p:
                output.append(self.parse_single(line))
            total_ans.append(output)
        return total_ans
                
            
        
    def parse_single(self,line):
        field = decode_instruction(line,self.field_order)
        op = self.op_dict[field['op'].item()]
        dst = self.register_dict[field['dst'].item()]
        src1 = self.register_dict[field['src1'].item()]
        src2 = self.register_dict[field['src2'].item()]
        
        imm = field['imm'].item()
        instruction = ' '.join([op, dst, src1, src2, str(imm)])
        return instruction
    
    
    def binary2risc(self, P):
        """
        P: (B,prog_len)
        """
        ans = []
        for p in P:
            program = []
            for line in p:
                instr = self.parse_single(line)
                program.append(instr)
            ans.append(program)
        return ans
    
    def risc2binary(self,lines):
        words = []
        for line in lines:
            tokens = line.strip().split()
            op_str, dst, src1, src2, imm = tokens
            op = self.op_dict2.get(op_str)

            dst = int(dst[1:])
            src1 = int(src1[1:])
            src2 = int(src2[1:])
            imm = int(imm)

            word = (op << (self.n_imm+self.n_src2+self.n_src1+self.n_dst)) | (dst << (self.n_imm+self.n_src2+self.n_src1)) | (src1 << (self.n_imm + self.n_src2)) | (src2 << self.n_imm) | imm
            words.append(torch.tensor(word))
        
        output = torch.stack(words).unsqueeze(0).unsqueeze(0)
        # print(output.shape)
        return output
        
    def assignRegister(self, src1_id, val1, src2_id, val2):
        registers = torch.zeros((1, 1, self.n_regs), dtype=torch.int)
        
        registers[..., src1_id] = val1
        registers[..., src2_id] = val2
        
        return registers
    
    def assignRegisterMlp(self, input):
        """
        input (B,2)
        """
        result = torch.sum(input,dim=1)
        registers = torch.zeros((input.size(0), 1, self.n_regs), dtype=torch.int)
        registers[:,0,1:3] = input
        registers[:,0,0] = result
        R = ((registers.unsqueeze(-1) >> torch.arange(self.n_val-1, -1, -1)) & 1).int()
        
        #print(R)
        return R
    
    def binary2register(self,R):
        """
        R (B,n_regs)
        """
        ans = []
        for r_set in R:
            R_set = []
            for i,r in enumerate(r_set):
                R_set.append(f"r{i}:{r}")
                
            ans.append(R_set)
        return ans
    
if __name__ == '__main__':
    parser = Parser(config)
    
    P = torch.tensor([[108,94,749,552]])
    p = parser.binary2risc(P)
    # print(p)
    lines = ["add r1 r2 r3 0","add r1 r1 r3 2","mul r3 r2 r3 1","mul r0 r2 r2 0"]
    program = parser.risc2binary(lines)
    print(f"program is {program}")
    registers = parser.assignRegister(2,3,3,4)
    R = ((registers.unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).int()
    #print(program)
    #print(R)
    """
    R = parser.assignRegisterMlp(torch.tensor([[1, 2],
                      [3, 4],
                      [1, 254]]))
                      """