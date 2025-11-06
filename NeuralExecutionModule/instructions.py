import torch
import torch.nn as nn
import torch.nn.functional as F


class Instruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, op1, op2):
        pass

class BitwiseOperator(Instruction):
    def __init__(self, config):
        super().__init__(config)
        self.k = self.config.k_bits
        self.l = config.input_l_bits # e.g 32 -> sign:1+ others:31
        
        self.register_buffer('add_table', self.create_add_table())
        self.register_buffer('sub_table', self.create_sub_table())
        self.register_buffer('carry_table', self.create_carry_table())
        self.register_buffer('borrow_table', self.create_borrow_table())
    
        
        self.register_buffer('add_lookup', F.one_hot(self.add_table.long(), num_classes=self.k).float())
        self.register_buffer('sub_lookup', F.one_hot(self.sub_table.long(), num_classes=self.k).float())
        self.register_buffer('carry_lookup', F.one_hot(self.carry_table.long(), num_classes=2).float())
        self.register_buffer('borrow_lookup', F.one_hot(self.borrow_table.long(), num_classes=2).float())
    
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
        B, l, k = op1.shape
        device = op1.device
        
        
        ##### TODO
        if self.add_lookup.device != device:
            self.add_lookup = self.add_lookup.to(device)
            self.sub_lookup = self.sub_lookup.to(device)
            self.carry_lookup = self.carry_lookup.to(device)
            self.borrow_lookup = self.borrow_lookup.to(device)
            
        add_result_dist = torch.zeros(B, l, k, device=device)
        sub_result_dist = torch.zeros(B, l, k, device=device)
        
        carry_dist = torch.zeros(B, 2, device=device)
        borrow_dist = torch.zeros(B, 2, device=device)
        carry_dist[:, 0] = 1.0
        borrow_dist[:, 0] = 1.0
        
        for i in range(l):
            add_prob_t = torch.einsum('bi,bj,bc->bijc', 
                                    op1[:, i], op2[:, i], carry_dist)
            sub_prob_t = torch.einsum('bi,bj,bc->bijc', 
                                    op1[:, i], op2[:, i], borrow_dist)
            
            add_result_dist[:, i] = torch.einsum('bijc,ijcv->bv', 
                                                add_prob_t, self.add_lookup)
            sub_result_dist[:, i] = torch.einsum('bijc,ijcv->bv', 
                                                sub_prob_t, self.sub_lookup)
            
            carry_dist = torch.einsum('bijc,ijcd->bd', 
                                    add_prob_t, self.carry_lookup)
            borrow_dist = torch.einsum('bijc,ijcd->bd', 
                                    sub_prob_t, self.borrow_lookup)
        
        return [add_result_dist, sub_result_dist]

    def to(self, device):
        super(BitwiseOperator, self).to(device)
        
        self.add_table = self.add_table.to(device)
        self.sub_table = self.sub_table.to(device)
        self.carry_table = self.carry_table.to(device)
        self.borrow_table = self.borrow_table.to(device)
        
        self.add_lookup = self.add_lookup.to(device)
        self.sub_lookup = self.sub_lookup.to(device)
        self.carry_lookup = self.carry_lookup.to(device)
        self.borrow_lookup = self.borrow_lookup.to(device)
        
        return self
    
class LogicOperator(Instruction):
    def __init__(self, config, operation):
        super().__init__(config)
        self.k = self.config.k_bits
        self.l = config.input_l_bits
        self.operation = operation
        
    def forward(self, op1, op2):
        if self.operation == 'and':
            result_dist = self.bitwise_and(op1, op2)
        elif self.operation == 'or':
            result_dist = self.bitwise_or(op1, op2)
        
        return result_dist
    
    def bitwise_and(self, op1, op2):
        """
        op1, op2 (b,l,k)
        """
        and_1 = op1[..., 1] * op2[..., 1]
        and_0 = 1 - and_1
        return torch.stack([and_0, and_1], dim=-1)
    
    def bitwise_or(self, op1, op2):
        """
        op1, op2 (b,l,k)
        """
        and_0 = op1[..., 0] * op2[..., 0]
        and_1 = 1 - and_0
        return torch.stack([and_0, and_1], dim=-1)

class Compare(Instruction):
    def __init__(self, config):
        super().__init__(config)
        self.k = self.config.k_bits
        self.l = config.input_l_bits
        
    def bitwise_compare(self, op1, op2):
        B = op1.shape[0]
        device = op1.device
        
        greater_prob = torch.zeros(B, device=device)
        less_prob = torch.zeros(B, device=device)
        equal_prob = torch.ones(B, device=device)
        
        
        for bit in reversed(range(self.l)):
            x_bit_0 = op1[:, bit, 0]  #(b,)
            x_bit_1 = 1 - x_bit_0
            y_bit_0 = op2[:, bit, 0]
            y_bit_1 = 1 - y_bit_0
            
            greater = x_bit_1 * y_bit_0
            less = x_bit_0 * y_bit_1
            

            new_greater = greater_prob + equal_prob * greater
            new_less = less_prob + equal_prob * less
            
            greater_prob = new_greater
            less_prob = new_less
            equal_prob = 1 - greater_prob - less_prob
            
        
        return greater_prob, equal_prob, less_prob
    
    def forward(self, op1, op2):
        greater_prob, equal_prob, less_prob = self.bitwise_compare(op1,op2)
        
        equal = equal_prob #(b,)
        not_equal = 1 - equal_prob
        greater = greater_prob
        less = less_prob
        greater_equal = greater_prob + equal_prob
        less_equal =  less_prob + equal_prob
        
        compare_feature = torch.stack([equal,not_equal,greater,less,greater_equal,less_equal],dim=1) #(b,6)
        
        return compare_feature
   
class Sin(Instruction):
    def __init__(self, config):
        super().__init__(config)
        self.k = self.config.k_bits
        self.l = config.input_l_bits 
        self.value_range = (0, 2*torch.pi)
    
    def forward(self, op1, op2):
        """
        op1 (b,l,k)
        """
        B = op1.shape[0]
        device = op1.device

        combined = torch.cat([op1, op2], dim=0) # (2b,l,k)
        expected_digits = torch.sum(combined * torch.arange(self.k, device=device).view(1, 1, self.k),dim=2)  # (2b, l)
        place_values = self.k ** torch.arange(0,self.l, device=device)
        expected_values = torch.sum(expected_digits * place_values, dim=1)  # (2b,)
        # print(f'expected values are {expected_values}')
        max_val = self.k ** self.l
        normalized_values = expected_values / max_val
        theta = self.value_range[0] + normalized_values * (self.value_range[1] - self.value_range[0])
        
       
        sin_values = torch.sin(1000*theta)  # (2b,)
        sin1, sin2 = sin_values[:B], sin_values[B:]

        sin_features = torch.stack([sin1, sin2], dim=1) #(b,2)
        return sin_features
    

class InstructionPool(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.k = config.k_bits
        self.l = config.input_l_bits

        self.bitwise = BitwiseOperator(config)
        self.logic_and = LogicOperator(config,'and')
        self.logic_or = LogicOperator(config,'or')
        self.compare = Compare(config)
        self.sin     = Sin(config)            

        self.compareproj = nn.Linear(6, self.l * self.k, bias=False)
        self.sinproj     = nn.Linear(2, self.l * self.k, bias=False)
    
    def forward(self, op1, op2):
        add_dist, sub_dist = self.bitwise(op1, op2)
        
        logic_and = self.logic_and(op1,op2)
        logic_or = self.logic_or(op1,op2)
        
        sin_feats = self.sin(op1,op2)
        sin = self.sinproj(sin_feats).reshape(op1.size(0), self.l, self.k)
        
        cmp_feats = self.compare(op1, op2)               
        cmp = self.compareproj(cmp_feats).reshape(op1.size(0), self.l, self.k)
        
        outs = [add_dist, sub_dist,logic_and,logic_or,sin,cmp]
        
        return torch.stack(outs,dim=1)
    
    
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
class Logictest(nn.Module):
    def __init__(self, k, l, operation):
        super().__init__()
        self.k = k
        self.l = l
        self.operation = operation
        
    def forward(self, op1, op2):
        if self.operation == 'and':
            result_dist = self.bitwise_and(op1, op2)
        elif self.operation == 'or':
            result_dist = self.bitwise_or(op1, op2)
        
        return result_dist
    
    def bitwise_and(self, op1, op2):
        """
        op1, op2 (b,l,k)
        """
        and_1 = op1[..., 1] * op2[..., 1]
        and_0 = 1 - and_1
        return torch.stack([and_0, and_1], dim=-1)
    
    def bitwise_or(self, op1, op2):
        """
        op1, op2 (b,l,k)
        """
        and_0 = op1[..., 0] * op2[..., 0]
        and_1 = 1 - and_0
        return torch.stack([and_0, and_1], dim=-1)
class Sintest(nn.Module):
    def __init__(self, l,k):
        super().__init__()
        self.k = k
        self.l = l
        self.value_range = (0, 2*torch.pi)
    
    def forward(self, op1, op2):
        """
        op1 (b,l,k)
        """
        B = op1.shape[0]
        device = op1.device

        combined = torch.cat([op1, op2], dim=0) # (2b,l,k)
        expected_digits = torch.sum(combined * torch.arange(self.k, device=device).view(1, 1, self.k),dim=2)  # (2b, l)
        print(f'expected digits are {expected_digits}')
        place_values = self.k ** torch.arange(0,self.l, device=device)
        expected_values = torch.sum(expected_digits * place_values, dim=1)  # (2b,)
        print(f'expected values are {expected_values}')
        max_val = self.k ** self.l
        normalized_values = expected_values / max_val
        theta = self.value_range[0] + normalized_values * (self.value_range[1] - self.value_range[0])
        
       
        sin_values = torch.sin(theta)  # (2b,)
        sin1, sin2 = sin_values[:B], sin_values[B:]

        sin_features = torch.stack([sin1, sin2], dim=1) #(b,2)
        return sin_features
class Comparetest(nn.Module):
    def __init__(self, l,k):
        super().__init__()
        self.k = k
        self.l = l
        
    def bitwise_compare(self, op1, op2):
        B = op1.shape[0]
        device = op1.device
        
        greater_prob = torch.zeros(B, device=device)
        less_prob = torch.zeros(B, device=device)
        equal_prob = torch.ones(B, device=device)
        
        
        for bit in reversed(range(self.l)):
            x_bit_0 = op1[:, bit, 0]  #(b,)
            x_bit_1 = 1 - x_bit_0
            y_bit_0 = op2[:, bit, 0]
            y_bit_1 = 1 - y_bit_0
            
            greater = x_bit_1 * y_bit_0
            less = x_bit_0 * y_bit_1
            

            new_greater = greater_prob + equal_prob * greater
            new_less = less_prob + equal_prob * less
            
            greater_prob = new_greater
            less_prob = new_less
            equal_prob = 1 - greater_prob - less_prob
            
        
        return greater_prob, equal_prob, less_prob
    
    def forward(self, op1, op2):
        greater_prob, equal_prob, less_prob = self.bitwise_compare(op1,op2)
        
        equal = equal_prob #(b,)
        not_equal = 1 - equal_prob
        greater = greater_prob
        less = less_prob
        greater_equal = greater_prob + equal_prob
        less_equal =  less_prob + equal_prob
        
        compare_feature = torch.stack([equal,not_equal,greater,less,greater_equal,less_equal],dim=1) #(b,6)
        
        return compare_feature
 
if __name__ == '__main__':
    a_dist = torch.zeros(1,3,2)
    b_dist = torch.zeros(1,3, 2)
    
    a_dist[0,0,0] = 1.0
    a_dist[0,1,0] = 1.0
    a_dist[0,2,1] = 1.0    #a=100_2 
    
    b_dist[0,0,0] = 1.0  
    b_dist[0,1,1] = 1.0  
    b_dist[0,2,0] = 1.0  # b=010_2
   
    
    add = Comparetest(3,2)
    
    addd= add(a_dist,b_dist)
    print(addd) 
    
        
        
        
        
        
        
        
    