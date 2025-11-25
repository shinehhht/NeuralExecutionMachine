import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class CrossAttention(nn.Module):
    def __init__(self, config, in_feature, out_feature):
        super().__init__()
        self.config = config
        
        
        self.k_proj = nn.Linear(in_feature, out_feature)
        self.v_proj = nn.Linear(in_feature, out_feature)

        self.dropout = nn.Dropout(config.dropout)
        self.o_proj = nn.Linear(out_feature, out_feature)
        
    
    def multihead_cross_attn(self, query, key, value, key_mask=None,query_mask=None):
        b,s_q,d = query.shape
        s_k = key.shape[1]
        head_dim = d // self.config.num_heads
        
        query = query.view(b, s_q, self.config.num_heads, head_dim).transpose(1, 2)
        key = key.view(b, s_k, self.config.num_heads, head_dim).transpose(1, 2)
        
       
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        #with open ("final_extra_hidden_info.txt",'a')as f:
                #f.write(f"original attn_scores {attn_scores.shape}\n{attn_scores[0]}\n")
                
        if key_mask is not None:
            mask = key_mask.squeeze(-1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, l]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            #with open ("final_extra_hidden_info.txt",'a')as f:
            #    f.write(f"attn_scores {attn_scores.shape}\n{attn_scores[0]}\n")
        
        
        if query_mask is not None:
            """
            mask = query_mask.unsqueeze(1).transpose(2, 3)  # [b, 1, 1, l]
            attn_scores = attn_scores.transpose(-2, -1).masked_fill(mask == 0, float('-inf'))
            attn_scores = attn_scores.transpose(-2, -1) 
            """
            qm = query_mask.squeeze(-1)            # [b, s_q]
            qm = qm.unsqueeze(1).unsqueeze(-1)     # [b, 1, s_q, 1]
            attn_scores = attn_scores.masked_fill(qm == 0, float('-inf'))

        
            
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        #with open ("final_extra_hidden_info.txt",'a')as f:
           #f.write(f"attn_weights {attn_weights.shape}\n{attn_weights[0]}\n")
        attn_weights = self.dropout(attn_weights)
        
        
        b,s_v,d = value.shape
        head_dim = d // self.config.num_heads
        value = value.view(b, s_v, self.config.num_heads, head_dim).transpose(1, 2)
        
        attn_output = torch.matmul(attn_weights, value)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, s_q, d)
        #with open ("final_extra_hidden_info.txt",'a')as f:
        #    f.write(f"attn_value {attn_output.shape}\n{attn_output[0]}\n")
        
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
        in_feature = config.nem_dim
        out_feature = config.hidden_dim
        super().__init__(config, in_feature, out_feature)
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, hidden_state, exec_outputs, mask):
        #with open ("final_extra_hidden_info.txt",'a')as f:
           #f.write("FUse2mAIN\n")
        answer_mask = ~mask
        #print(answer_mask[0])
        K = self.k_proj(exec_outputs) #(b,r,d)
        V = self.v_proj(exec_outputs) #(b,r,d)
        Q = self.q_proj(hidden_state) #(b,l,d)
        
        #with open ("final_extra_hidden_info.txt",'a')as f:
            #f.write(f'exec shape {exec_outputs.shape}, exec output: {exec_outputs[0]}')
            #f.write(f"K {K[0]}\n")
            #f.write(f"Q {Q[0]}\n")
            
        #attn_output = self.multihead_cross_attn(Q, K, V, query_mask=answer_mask)
        attn_output = self.multihead_cross_attn(Q, K, V)
        
        
        #with open ("final_extra_hidden_info.txt",'a')as f:
            #f.write(f"final shape is {attn_output.shape}\n")
            #f.write(str(attn_output[0]))

        fused_states = hidden_state + attn_output
        
        return fused_states
    
    
class MultiHeadSlotSelfAttn(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Z):
        B, S, D = Z.shape
        H = self.num_heads
        Hd = self.head_dim

        Q = self.q_proj(Z)  # (B,S,D)
        K = self.k_proj(Z)
        V = self.v_proj(Z)

        Q = Q.view(B, S, H, Hd).transpose(1, 2)  # (B,H,S,Hd)
        K = K.view(B, S, H, Hd).transpose(1, 2)
        V = V.view(B, S, H, Hd).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.dropout(attn_weights)


        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.out_proj(out)
        return out

class SlotFullBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.nem_dim

        self.self_attn = MultiHeadSlotSelfAttn(
            dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.ln1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, Z):
        Z_res = Z
        Z_attn = self.self_attn(Z)
        Z = self.ln1(Z_res + Z_attn)

        Z_res = Z
        Z_ffn = self.ffn(Z)
        Z = self.ln2(Z_res + Z_ffn)

        return Z
    
class TinySlotSelfAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_Q = nn.Linear(dim, dim)
        self.proj_K = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, Z, mask=None):
        Q = self.proj_Q(Z)
        K = self.proj_K(Z)
        V = Z

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out
    
class SlotTinyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.nem_dim

        self.self_attn = TinySlotSelfAttn(dim=dim)
        self.ln1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, Z):
        Z_res = Z
        Z_attn = self.self_attn(Z)
        Z = self.ln1(Z_res + Z_attn)

        Z_res = Z
        Z_ffn = self.ffn(Z)
        Z = self.ln2(Z_res + Z_ffn)

        return Z
       
class InitialParse(CrossAttention):
    def __init__(self,config):
        in_feature = config.hidden_dim
        out_feature = config.nem_dim
        super().__init__(config, in_feature, out_feature)
         
        # Q
        total_slots_with_registers = 2 + 5 * config.num_instructions + 1
        self.slot_queries = nn.Parameter(torch.randn(total_slots_with_registers, config.nem_dim))
        #self.slot_attn = SlotTinyBlock(config)
        self.ln = nn.LayerNorm(config.nem_dim)
        
    def forward(self, hidden_state, mask):
        """
        hidden_state (b,l,d)
        """
        batch_size = hidden_state.shape[0]
        K = self.k_proj(hidden_state)  # (b, l, d)
        V = self.v_proj(hidden_state)
        
        Q = self.slot_queries.unsqueeze(0).expand(batch_size,-1,-1) # (b,slots,d)
        
        #Z = self.multihead_cross_attn(Q, K, V, key_mask=mask)
        #Z = self.slot_attn(Z) 
        
        Z = self.ln(self.multihead_cross_attn(Q, K, V, key_mask=mask))
        
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