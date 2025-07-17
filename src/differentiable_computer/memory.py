import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, address_num, value_size):
        super().__init__()
        self.address_num = address_num
        self.value_size = value_size
        self.memory = nn.Parameter(torch.zeros(address_num, value_size)) #  the probability of m_t^i taking the value j
    
    def read(self,address):
        """
        address: (batch, address_num)
        """
        value = torch.matmul(address, self.memory) # 所有address行的每列加权和
        return value
    
    def write(self, address, content, write_prob):
        """
        address: (batch, address_num)
        content: (batch, value_dim)
        """
        batch_size = address.size(0)
        old_mem = self.memory.unsqueeze(0).repeat(batch_size,1,1)
        # print(old_mem.shape)
        M_write = (1-address.unsqueeze(2)) * old_mem + address.unsqueeze(2) * content.unsqueeze(1)
        # print(M_write.shape)
        new_mem = (1-write_prob) * old_mem + write_prob * M_write
        
        self.memory.data = new_mem.mean(dim=0)
        
class Register(nn.Module):
    def __init__(self, register_num, value_size):
        super().__init__()
        self.register_num = register_num
        self.value_dim = value_size
        self.registers = nn.Parameter(torch.zeros(register_num, value_size))   
        
    def read(self,address):
        """
        address: (batch, register_num)
        """
        value = torch.matmul(address, self.registers) 
        return value
    
    def write(self, address, content, write_prob):
        batch_size = address.size(0)
        old_reg = self.registers.unsqueeze(0).repeat(batch_size, 1, 1)
        reg_write = (1 - address.unsqueeze(2)) * old_reg + address.unsqueeze(2) * content.unsqueeze(1)
        
        new_reg = (1-write_prob) * old_reg + write_prob * reg_write
        self.registers.data = new_reg.mean(dim=0)
        
    def update_concurrent(self, p_dst, value_list, pw_list):
        """
        p_dst : (B,n_regs)
        value_list: [(B,value_size), (B,value_size), ...]
        pw_list: [(B,n_regs,value_size), (B,n_regs,value_size), ...]
        
        value已经是各instruction的加权value
        write_prob为总的写入概率
        """
        print(f"pw_list {pw_list}")
        print(f"value list is {value_list}")
        
        w_stack = torch.stack(pw_list)            # (N,B,1)
        combined = torch.stack(value_list).sum(0) 
        print(f"combined {combined}")
        w_sum = w_stack.sum(0) # (B,1)
        
        write_prob = p_dst.unsqueeze(-1) * w_sum.unsqueeze(1) 
        print(f"write_prob {write_prob}")
        self.write(p_dst, combined, write_prob)

        
        
    
if __name__ == '__main__':
    test = Memory(3,5)
    address = torch.zeros((3,3))
    content = torch.zeros((3,5))
    write_prob = 0.3
    test.write(address, content, write_prob)