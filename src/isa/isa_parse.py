import torch
"""
op | dst | src1 | src2 | imm

op: 4 bit
dst: 3 bit
src1: 3 bit
src2: 3 bit
imm: 3 bit

OP_SET = [
    'add',      0000b
    'sub',      0001b
    'mul',      0010b
    'cmp_eq',   0011b
    'cmp_lt',   0100b
    'cmp_gt',   0101b
    'jmp',      0110b
    'halt'      0111b
    'load',     1000b
    'store',    1001b
    'call':     1010b,    
    'ret':      1011b,
    ]
"""
def decimal2binary(n):
    return format(n, '016b')

class Parser():
    def __init__(self):
        super().__init__()
        self.op_dict = {
            0:'add',
            1:'sub',
            2:'mul',
            3:'cmp_eq',
            4:'cmp_lt',
            5:'cmp_gt',
            6:'jmp',
            7:'halt',
            8:'load',
            9:'store',
            10:'call',
            11:'ret' ,
            12:'mv',
            13:'and',
            14:'or',
            15:'not'
        }
        
        self.op_dict2 = {
            "add": 0,
            "sub": 1,
            "mul": 2,
            "cmp_eq": 3,
            "cmp_lt": 4,
            "cmp_gt": 5,
            "jmp": 6,
            "mv": 12,
        }
        
        
        self.register_dict = {
            0:'r0',
            1:'r1',
            2:'r2',
            3:'r3',
            4:'r4',
            5:'r5',
            6:'r6',
            7:'r7',
            8:'r8'
        }
        
        
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
        op = self.op_dict[((line >> 12) & 0xF).item()]
        dst = self.register_dict[((line >> 9) & 0x7).item()]
        src1 = self.register_dict[((line >> 6) & 0x7).item()]
        src2 = self.register_dict[((line >> 3) & 0x7).item()]
        
        imm = (line & 0x7).item() - 2**2
        instruction = ' '.join([op, dst, src1, src2, str(imm)])
        return instruction
    
    def binary2risc(self, P):
        """
        P: (B,L,prog_len)
        """
        ans = []
        for p in P:
            program = []
            for line in p:
                op = self.op_dict[((line >> 12) & 0xF).item()]
                dst = self.register_dict[((line >> 9) & 0x7).item()]
                src1 = self.register_dict[((line >> 6) & 0x7).item()]
                src2 = self.register_dict[((line >> 3) & 0x7).item()]
                
                imm = (line & 0x7).item() - 2**2
                instruction = ' '.join([op, dst, src1, src2, str(imm)])
                program.append(instruction)
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

            word = (op << 12) | (dst << 9) | (src1 << 6) | (src2 << 3) | imm
            words.append(torch.tensor(word))
        
        output = torch.stack(words).unsqueeze(0).unsqueeze(0)
        print(output)
        return output
        
    def assignRegister(self, src1_id, val1, src2_id, val2):
        registers = torch.zeros((1, 1, 8), dtype=torch.float32)
        
        registers[..., src1_id] = val1
        registers[..., src2_id] = val2
        
        return registers
        
    
if __name__ == '__main__':
    parser = Parser()
    P = torch.tensor([33181,16478,13037,27176])
    lines = ["add r1 r2 r3 0", "sub r0 r1 r2 1"]
    program = parser.risc2binary(lines)
    registers = parser.assignRegister(2,3,3,4)
    print(program)
    print(registers)