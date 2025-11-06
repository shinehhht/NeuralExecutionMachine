import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    
    def multihead_cross_attn(self, query, key, value):
        b,s_q,d = query.shape
        s_k = key.shape[1]
        head_dim = d // self.config.num_heads
        
        query = query.view(b, s_q, self.config.num_heads, head_dim).transpose(1, 2)
        key = key.view(b, s_k, self.config.num_heads, head_dim).transpose(1, 2)
        
       
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        
        b,s_v,d = value.shape
        head_dim = d // self.config.num_heads
        value = value.view(b, s_v, self.config.num_heads, head_dim).transpose(1, 2)
        
        attn_output = torch.matmul(attn_weights, value)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, s_q, d)
        attn_output = self.o_proj(attn_output)
       
        return attn_output
    
    
    def forward(self):
        pass
        
        
class Transformer2Prog(CrossAttention):
    def __init__(self,config):
        super().__init__(config)
         
        # Q
        self.slot_queries = nn.Parameter(torch.randn(config.total_slots, config.hidden_dim))
        
    def forward(self, hidden_state):
        """
        hidden_state (b,l,d)
        
        """
        assert self.config.total_slots == 2 * self.config.num_instructions + self.config.num_op + self.config.num_op * self.config.num_instructions
        batch_size = hidden_state.shape[0]
        
        K = self.k_proj(hidden_state)  # (b, l, d)
        V = self.v_proj(hidden_state)
        
        Q = self.slot_queries.unsqueeze(0).expand(batch_size,-1,-1) # (b,slots,d)
        
        Z = self.multihead_cross_attn(Q,K,V) # (b,slots,d)
        
        
        instruction_tokens = self.config.num_instructions
        op_tokens = self.config.num_op
        mask_tokens = self.config.num_op*self.config.num_instructions
    
        opcode = Z[:, :instruction_tokens, :]
        op = Z[:, instruction_tokens:instruction_tokens+op_tokens, :]
        mask = Z[:, instruction_tokens+op_tokens:instruction_tokens+op_tokens+mask_tokens, :]
        cond = Z[:, -instruction_tokens:, :]
        
        return opcode, op, mask, cond       


class Prog2Transformer(CrossAttention):
    def __init__(self,config):
        super().__init__(config)
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, hidden_state, exec_outputs):
        K = self.k_proj(exec_outputs) #(b,k,d)
        V = self.v_proj(exec_outputs) #(b,k,d)
        
        Q = self.q_proj(hidden_state) #(b,l,d)

        attn_output = self.multihead_cross_attn(Q, K, V)
        
        fused_states = hidden_state + attn_output
        
        return fused_states
    
class InitialParse(CrossAttention):
    def __init__(self,config):
        super().__init__(config)
         
        # Q
        total_slots_with_registers = 2 + 5 * config.num_instructions + 1
        self.slot_queries = nn.Parameter(torch.randn(total_slots_with_registers, config.hidden_dim))
        self.ln = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, hidden_state):
        """
        hidden_state (b,l,d)
        """
        batch_size = hidden_state.shape[0]
        K = self.k_proj(hidden_state)  # (b, l, d)
        V = self.v_proj(hidden_state)
        
        Q = self.slot_queries.unsqueeze(0).expand(batch_size,-1,-1) # (b,slots,d)
        
        Z = self.ln(self.multihead_cross_attn(Q,K,V)) # (b,slots,d)
        
        
        instruction_tokens = self.config.num_instructions
    
        # for initialize registers
        value_initial = Z[:, 0:1, :]
        k_initial = Z[:, 1:2, :]
        final_registers_gate = Z[:,2,:]
        
        gate = Z[:,3:3+instruction_tokens, :]
        k_write = Z[:, 3+instruction_tokens:3+2*instruction_tokens, :]
        q_read = Z[:, 3+2*instruction_tokens:3+4*instruction_tokens, :]
        opcode = Z[:, 3+4*instruction_tokens:3+5*instruction_tokens, :]
        #cond = Z[:, -instruction_tokens:, :]
        
        return final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode#, cond