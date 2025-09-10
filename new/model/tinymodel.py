import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from NEM.nem import NeuralExecutionModule, NeuralExecutionModuleWithRegisters


class SimpleTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            self.num_heads, 
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout_rate)
        )
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, x, attention_mask=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        
        output: (batch_size, seq_len, hidden_dim)
        """
        
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + attn_output
        
        residual = x
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output
        
        return x

class ModelWithoutNEM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.interprate = nn.Linear(1,config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 19)
        
        self.transformer_block1 = SimpleTransformerBlock(config)
        self.transformer_block2 = SimpleTransformerBlock(config)
        self.transformer_block3 = SimpleTransformerBlock(config)

    def forward(self,x):
        """
        x (b,2,1)
        """
        parse2hidden = self.interprate(x)
        
        hidden_state1 = self.transformer_block1(parse2hidden)
        hidden_state2 = self.transformer_block2(hidden_state1)
        hidden_state3 = self.transformer_block3(hidden_state2)
       
        output = self.output(hidden_state3[:,-1,:]) # (b,19)
        
        return output 
        



class MiniModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.interprate = nn.Linear(1,config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 19)
        
        self.NEM = NeuralExecutionModuleWithRegisters(config)
        self.transformer_block1 = SimpleTransformerBlock(config)
        self.transformer_block2 = SimpleTransformerBlock(config)
        self.transformer_block3 = SimpleTransformerBlock(config)

        
    def forward(self,x):
        """
        x (b,2,1)
        """
        parse2hidden = self.interprate(x)
        
        hidden_state1 = self.transformer_block1(parse2hidden)
        #print(f'hidden 1 shape {hidden_state1.shape}')
        fused_hidden_states1 = self.NEM(hidden_state1)
        
        hidden_state2 = self.transformer_block2(fused_hidden_states1)
        fused_hidden_states2 = self.NEM(hidden_state2)
        
        hidden_state3 = self.transformer_block3(fused_hidden_states2)
        fused_hidden_states3 = self.NEM(hidden_state3) # (b,2,d)
        #print(f'fused hidden state {fused_hidden_states3.shape}')
        output = self.output(fused_hidden_states3[:,-1,:]) # (b,19)
        
        return output 
        