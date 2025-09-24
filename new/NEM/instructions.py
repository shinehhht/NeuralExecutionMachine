import torch
import torch.nn as nn
import torch.nn.functional as F


class Instruction(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        
    def forward(self, op1, op2):
        pass
    
class Add(Instruction):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.k = self.config.k_bits
        self.l = args.input_l_bits # e.g 32 -> sign:1+ others:31
        
        self.register_buffer('add_table', self.create_add_table())
        self.register_buffer('carry_table', self.create_carry_table())
        
    def create_add_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for a in range(self.k):
            for b in range(self.k):
                for c in range(2):
                    table[a, b, c] = (a + b + c) % self.k
        return table
    
    def create_carry_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for a in range(self.k):
            for b in range(self.k):
                for c in range(2):
                    table[a, b, c] = (a + b + c) // self.k
        return table
        
    def forward(self, op1, op2):
        """
        op1, op2 (b,l,k)
        """
        B,l,k = op1.shape
        device = op1.device
        assert l == self.l
        assert k == self.k
        
        result_dist = torch.zeros(B, l, k, device=device)
        carry_dist = torch.zeros(B, 2, device=device)
        carry_dist[:, 0] = 1.0
        
        for i in range(l):
            prob_t = (op1[:, i, :, None, None] * op2[:, i, None, :, None] * carry_dist[:, None, None, :]) #(b, k, k, 2)
        
            for v in range(k):
                mask = (self.add_table == v)
                result_dist[:, i, v] = prob_t[:, mask].sum(dim=1)
            
            new_carry = torch.zeros(B, k, device=device)
            for carry in range(2):
                mask = (self.carry_table == carry) #(k,k,2)
                new_carry[:,carry] = prob_t[:, mask].sum(dim=1)
            
            carry_dist = new_carry 
        
        # with carry
        #padded_carry = F.pad(carry_dist, (0, k-2))
        #result_dist[:, l] = padded_carry
        
        return result_dist
    
    
class Sub(Instruction):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.k = self.config.k_bits
        self.l = args.input_l_bits
        
        self.register_buffer('sub_table', self.create_sub_table())
        self.register_buffer('borrow_table', self.create_borrow_table())
        
    def create_sub_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for x in range(self.k):
            for y in range(self.k):
                for b in range(2):
                    o = x - y - b
                    if o >= 0:
                        table[x, y, b] = o
                    else:
                        table[x, y, b] = self.k + o
        return table
    
    def create_borrow_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for x in range(self.k):
            for y in range(self.k):
                for b in range(2):
                    diff = x - y - b
                    table[x, y, b] = 1 if diff < 0 else 0
        return table
    
    def forward(self, op1, op2):
        B,l,k = op1.shape
        device = op1.device
        assert l == self.l
        assert k == self.k
        
        result_dist = torch.zeros(B, l, k, device=device)
        borrow_dist = torch.zeros(B, 2, device=device)
        borrow_dist[:, 0] = 1.0
        
        for i in range(l):
            prob_t = (op1[:, i, :, None, None] * op2[:, i, None, :, None] * borrow_dist[:, None, None, :]) #(b, k, k, 2)
            
            for v in range(k):
                mask = (self.sub_table == v)
                result_dist[:, i, v] = prob_t[:, mask].sum(dim=1)
            
            new_borrow = torch.zeros(B, 2, device=device)
            for borrow in range(2):
                mask = (self.borrow_table == borrow) # (k,k,2)
                new_borrow[:, borrow] = prob_t[:, mask].sum(dim=1)
            
            borrow_dist = new_borrow
        
        
        # with borrow
        # padded_borrow = F.pad(borrow_dist, (0, k-2))
        # result_dist[:, l] = padded_borrow
        return result_dist
    

class Floor(Instruction):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.k = self.config.k_bits
        self.l = args.input_l_bits
        
    
class Subtest(nn.Module):
    def __init__(self, k,l):
        super().__init__()
        self.k = k
        self.l = l
        
        self.register_buffer('sub_table', self.create_sub_table())
        self.register_buffer('borrow_table', self.create_borrow_table())
        
    def create_sub_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for x in range(self.k):
            for y in range(self.k):
                for b in range(2):
                    o = x - y - b
                    if o >= 0:
                        table[x, y, b] = o
                    else:
                        table[x, y, b] = self.k + o
        return table
    
    def create_borrow_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for x in range(self.k):
            for y in range(self.k):
                for b in range(2):
                    diff = x - y - b
                    table[x, y, b] = 1 if diff < 0 else 0
        return table
    
    def forward(self, op1, op2):
        B,l,k = op1.shape
        assert l == self.l
        assert k == self.k
        
        result_dist = torch.zeros(B, l, k)
        borrow_dist = torch.zeros(B, 2)
        borrow_dist[:, 0] = 1.0
        
        for i in range(l):
            prob_t = (op1[:, i, :, None, None] * op2[:, i, None, :, None] * borrow_dist[:, None, None, :]) #(b, k, k, 2)
            
            for v in range(k):
                mask = (self.sub_table == v)
                result_dist[:, i, v] = prob_t[:, mask].sum(dim=1)
            
            new_borrow = torch.zeros(B, 2)
            for borrow in range(2):
                mask = (self.borrow_table == borrow) # (k,k,2)
                new_borrow[:, borrow] = prob_t[:, mask].sum(dim=1)
            
            borrow_dist = new_borrow
        
       
        return result_dist  
class Addtest(nn.Module):
    def __init__(self, k,l):
        super().__init__()
        self.k = k
        self.l = l
        
        self.register_buffer('add_table', self.create_add_table())
        self.register_buffer('carry_table', self.create_carry_table())
        
    def create_add_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for a in range(self.k):
            for b in range(self.k):
                for c in range(2):
                    table[a, b, c] = (a + b + c) % self.k
        return table
    
    def create_carry_table(self):
        table = torch.zeros(self.k, self.k, 2)
        for a in range(self.k):
            for b in range(self.k):
                for c in range(2):
                    table[a, b, c] = (a + b + c) // self.k
        return table
        
    def forward(self, op1, op2):
        """
        op1, op2 (b,l,k)
        """
        B,l,k = op1.shape
        assert l == self.l
        assert k == self.k
        
        result_dist = torch.zeros(B, l, k)
        carry_dist = torch.zeros(B, 2)
        carry_dist[:, 0] = 1.0
        
        for i in range(l):
            prob_t = (op1[:, i, :, None, None] * op2[:, i, None, :, None] * carry_dist[:, None, None, :]) #(b, k, k, 2)
        
            for v in range(k):
                mask = (self.add_table == v)
                result_dist[:, i, v] = prob_t[:, mask].sum(dim=1)
            
            new_carry = torch.zeros(B, k)
            for carry in range(2):
                mask = (self.carry_table == carry) #(k,k,2)
                new_carry[:,carry] = prob_t[:, mask].sum(dim=1)
            
            carry_dist = new_carry 
        
        return result_dist
    
if __name__ == '__main__':
    a_dist = torch.zeros(1,5, 2)
    b_dist = torch.zeros(1,5, 2)
    
    a_dist[0,0,0] = 1.0
    a_dist[0,1,1] = 1.0  
    a_dist[0,2,0] = 1.0  
    a_dist[0,3,0] = 1.0
    a_dist[0,4,1] = 1.0 # a=10010_2 
    
    b_dist[0,0,1] = 1.0  
    b_dist[0,1,0] = 1.0  
    b_dist[0,2,1] = 1.0  # b=01101_2
    b_dist[0,3,1] = 1.0  
    b_dist[0,4,0] = 1.0  
    
    add = Addtest(2,5)
    
    out = add(a_dist,b_dist)
    print(out)        
        
        
        
        
        
        
        
    