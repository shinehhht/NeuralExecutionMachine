import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        
        self.group_in_features = in_features // groups
        self.group_out_features = out_features // groups
        
        self.weight = nn.Parameter(torch.empty(groups, self.group_out_features, self.group_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        for g in range(self.groups):
            nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.group_in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        B, L, _ = input.shape
        
        input_grouped = input.view(B, L, self.groups, self.group_in_features)
        output_grouped = torch.einsum('blgi,goi->blgo', input_grouped, self.weight)
        
        output = output_grouped.contiguous().view(B, L, self.out_features)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output


class GroupedCrossAttention(nn.Module):
    def __init__(self, config, groups=4):
        super().__init__()
        self.config = config
        self.groups = groups
        
        self.k_proj = GroupedLinear(config.hidden_dim, config.hidden_dim, groups=groups)
        self.v_proj = GroupedLinear(config.hidden_dim, config.hidden_dim, groups=groups)
        self.o_proj = GroupedLinear(config.hidden_dim, config.hidden_dim, groups=groups)
        
        self.dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
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


class Prog2Transformer(GroupedCrossAttention):
    def __init__(self,config, groups=4):
        super().__init__(config)
        
        self.q_proj = GroupedLinear(config.hidden_dim, config.hidden_dim, groups=groups)
        
    def forward(self, hidden_state, exec_outputs):
        K = self.k_proj(exec_outputs) #(b,k,d)
        V = self.v_proj(exec_outputs) #(b,k,d)
        
        Q = self.q_proj(hidden_state) #(b,l,d)

        attn_output = self.multihead_cross_attn(Q, K, V)
        
        fused_states = hidden_state + attn_output
        
        return fused_states
        



class InitialParseGrouped(GroupedCrossAttention):
    def __init__(self, config, groups=4):
        super().__init__(config, groups)
        
        
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
        
        Q = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (b,slots,d)

        Z = self.ln(self.multihead_cross_attn(Q, K, V))  # (b,slots,d)
        
        instruction_tokens = self.config.num_instructions
        
        # 解析输出
        value_initial = Z[:, 0:1, :]
        k_initial = Z[:, 1:2, :]
        final_registers_gate = Z[:, 2, :]
        
        gate = Z[:, 3:3+instruction_tokens, :]
        k_write = Z[:, 3+instruction_tokens:3+2*instruction_tokens, :]
        q_read = Z[:, 3+2*instruction_tokens:3+4*instruction_tokens, :]
        opcode = Z[:, 3+4*instruction_tokens:3+5*instruction_tokens, :]
        
        return final_registers_gate, value_initial, k_initial, gate, k_write, q_read, opcode
