import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from NEM.group_nem import NeuralExecutionModuleWithSharedModule
from NEM.featurePproj import FeatureProjWithRegisters
from NEM.interpreter import InterpreterWithRegistersAndKbit

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
    def __init__(self,config,args):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, args.output_bits*10)
        
        self.transformer_block1 = SimpleTransformerBlock(config)
        self.transformer_block2 = SimpleTransformerBlock(config)
        self.transformer_block3 = SimpleTransformerBlock(config)

    def forward(self,x):
        """
        x (b,2)
        """
        b = x.size(0)
        parse2hidden = self.embedding(x)
        # print(f'hidden shape {parse2hidden.shape}')
        hidden_state1 = self.transformer_block1(parse2hidden)
        hidden_state2 = self.transformer_block2(hidden_state1)
        hidden_state3 = self.transformer_block3(hidden_state2)
       
        output = self.output(hidden_state3[:,-1,:]).view(b,-1,10)
        
        return output 
        



class MiniModelShared(nn.Module):
    def __init__(self,config,args):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, args.output_bits*10)
        self.FeatureProj = FeatureProjWithRegisters(config)
        self.Interpreter = InterpreterWithRegistersAndKbit(config,args)
        
        self.NEM1 = NeuralExecutionModuleWithSharedModule(config,args,self.FeatureProj,self.Interpreter)
        self.NEM2 = NeuralExecutionModuleWithSharedModule(config,args,self.FeatureProj,self.Interpreter)
        self.NEM3 = NeuralExecutionModuleWithSharedModule(config,args,self.FeatureProj,self.Interpreter)
        
        self.transformer_block1 = SimpleTransformerBlock(config)
        self.transformer_block2 = SimpleTransformerBlock(config)
        self.transformer_block3 = SimpleTransformerBlock(config)

        
    def forward(self,x,epoch):
        """
        x (b,10)
        """
        
        b = x.size(0)
        parse2hidden = self.embedding(x)
        
        hidden_state1 = self.transformer_block1(parse2hidden)
        fused_hidden_states1,_,_,program_details1 = self.NEM1(hidden_state1,epoch)
        
        hidden_state2 = self.transformer_block2(fused_hidden_states1)
        fused_hidden_states2,_,_,program_details2 = self.NEM2(hidden_state2,epoch)
        
        hidden_state3 = self.transformer_block3(fused_hidden_states2)
        fused_hidden_states3, origin_gate, processed_gate,program_details3 = self.NEM3(hidden_state3,epoch) # (b,10,d)
        
        
        #print(f'fused hidden state {fused_hidden_states3.shape}')
        output = self.output(fused_hidden_states3[:,-1,:]).view(b,-1,10) # (b,bits,10)
        program = {
            'DATA':x[0],
            'NEM1':program_details1,
            'NEM2':program_details2,
            'NEM3':program_details3
        }
        return output, origin_gate, processed_gate, program
    
    def generate(self,x):
        b = x.size(0)
        parse2hidden = self.embedding(x)
        
        hidden_state1 = self.transformer_block1(parse2hidden)
        fused_hidden_states1 = self.NEM1.generate(hidden_state1)
        
        hidden_state2 = self.transformer_block2(fused_hidden_states1)
        fused_hidden_states2 = self.NEM2.generate(hidden_state2)
        
        hidden_state3 = self.transformer_block3(fused_hidden_states2)
        fused_hidden_states3 = self.NEM3.generate(hidden_state3) # (b,2,d)
        #print(f'fused hidden state {fused_hidden_states3.shape}')
        output = self.output(fused_hidden_states3[:,-1,:]).view(b,-1,10)
        
        return output 