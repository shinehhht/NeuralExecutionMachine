import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from config import config
from isa.isa_parse import Parser
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from llama_model.nem import NeuralExecutionModule
from utils import assistLoss


    
class Augmented_LLM(nn.Module):
    def __init__(self, LLM, Tokenizer,n_regs, n_val, prog_max_length, dropout, hidden_dim, max_tokens, intermidate_dim, temp=0.85, top_k=50):
        super().__init__()
        self.LLM = LLM
        self.tokenizer = Tokenizer
        self.max_tokens = max_tokens
        self.temp = temp
        self.top_k = top_k
        
        self.zLayernorm   = nn.LayerNorm(hidden_dim, eps=1e-5)
        
        self.NEM = NeuralExecutionModule(hidden_dim,intermidate_dim,n_regs,n_val,prog_max_length,dropout)
         
    def forward(self, input_ids, attention_mask, labels, group_num=10):
        """
        text -> LLM -> z_hidden -> R M P P_len -> interpreter -> final R M -> computed-state -> fusion -> lm-head -> output
        """
        
        outputs = self.LLM(input_ids, attention_mask=attention_mask, output_hidden_states = True)
     
        z_hidden = outputs.hidden_states[-1] # (B,L,H)
        z_hidden = self.zLayernorm(z_hidden)
        
        results = self.NEM(z_hidden)

        threshold = results['threshold']
        reg_hidden = results['reg_hidden']
        candidate_set = results['candidate_set']
        current_set = results['current_set']
        
        final_hidden = (1-threshold.unsqueeze(-1))* z_hidden.unsqueeze(2) + threshold.unsqueeze(-1) * reg_hidden
        logits = self.LLM.lm_head(final_hidden) #(B,L,group,V)
        
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        shift_labels = shift_labels.unsqueeze(2).expand(-1, -1, len(candidate_set))
        
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction='none'
        ).view(logits.size(0), -1, logits.size(2)) # (B,L-1,group)

        loss = loss.sum(dim=1).sum(dim=0)  # (group，)

        best_idx = torch.argmin(loss,dim=-1).item()
        #print(f"best idx is {best_idx}, loss is {loss[best_idx]}")
        best_candidate =  candidate_set[best_idx]
        assist_loss = assistLoss(current_set, best_candidate)
        final_loss = loss[best_idx]
        
        total_loss = 0.8 * final_loss + 0.2 * assist_loss
        
        #print(f"total loss is {total_loss}")
        return {"logits": logits, "loss":total_loss, "assist loss":assist_loss}
        
        
        """
        for i in range(group_num):
            R_bit = candidate_set[i]['R'] # (B,L,n_regs,16)
            logits_bit = candidate_set[i]['logits'] 
            prog_len_bit = candidate_set[i]['prog_len'] # (B,L,4)
            
            R = R_bit2decimal(R_bit) # (B,L,n_regs)
            M = torch.zeros_like(R)
            prog_len = bit2decimal(prog_len_bit)
            P = program_bit2decimal(logits_bit)
            
            # print(f"program is {P.shape}")
            interpreter = Interpreter(R, M, P, prog_len)
            
            registers,_ = interpreter.run() # (B,L,n_regs)


            reg_proj = self.register2hidden(registers)
            # reg_hidden = reg_proj / (reg_proj.norm(dim=-1, keepdim=True) + 1e-6)
            # print(f"std is {reg_hidden.std()}")
            reg_hidden = self.regLayernorm(reg_proj)
                
            final_hidden = (1-threshold)* z_hidden + threshold * reg_hidden
            logits = self.LLM.lm_head(final_hidden)
       
       
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # logits (B*L,V).  labels (B*L,)
                loss_list.append(loss)
        
              
        loss_tensor = torch.stack(loss_list, dim=0)
        print(f"loss shape {loss_tensor.shape}")
        best_idx = torch.argmin(loss_tensor).item()
        print(f"best idx is {best_idx}, loss is {loss_list[best_idx]}")
        best_candidate = candidate_set[best_idx]
        
        assist_loss = assistLoss(current_set, best_candidate)
        final_loss = loss_list[best_idx]
        
        total_loss = 0.8 * final_loss + 0.2 * assist_loss
        
        return {"logits": logits, "loss":total_loss}
        """
    
    """
    Done after finish training.
    def generate(self, text):
        dir_path = Path(__file__).parent
        record_path = dir_path / "record_program.txt"
        input_ids = self.tokenizer.encode(text,return_tensors="pt").to('cuda:0')

        generated_tokens = []
        with open(record_path,"a") as f:
            f.write("Begin inference\n")
        for i in range(self.max_tokens):
            with open(record_path,"a") as f:
                f.write(f"\n\n\n{i}th token generation\n")
                
            outputs = self.LLM(input_ids, output_hidden_states = True)
            z_hidden = outputs.hidden_states[-1][:,-1,:].unsqueeze(1) # (B,1,hidden_dim)
            z_hidden = self.zLayernorm(z_hidden)
    
            threshold, P, R, M, prog_len = self.aug_layer(z_hidden) # (B,1,prog_len)
            print(f"Before interpreter\n\n program {P.requires_grad} register {R.requires_grad} length {prog_len.requires_grad}")
        
            parser = Parser()
            program_list = parser.parse(P)
            with open(record_path,"a") as f:
                for idx, prog in enumerate(program_list):
                    f.write(f"{idx} Program is {prog}\n Program length is {torch.argmax(prog_len,dim=-1)}")
            interpreter = Interpreter(R, M, P, prog_len)
            registers,_ = interpreter.run() 
            # exec_state = torch.cat((self.register2hidden(registers),self.mem2hidden(memory)),dim=-1) # (B, hidden_state)
            print(f"After interpreter\n register {registers.requires_grad}")
            reg_hidden = self.register2hidden(registers)
            reg_hidden = self.regLayernorm(reg_hidden)
            # final_hidden = z_hidden + exec_state
            print(f"threshold is {threshold}")
            final_hidden = (1-threshold)* z_hidden + threshold * reg_hidden
            logits = self.LLM.lm_head(final_hidden).squeeze(1)

            logits = logits / self.temp # (1,vocab_size)
            if self.top_k is not None:
                top_k = min(self.top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, top_k)
                min_topk = topk_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_topk, torch.full_like(logits, -float("inf")), logits)
            
            probs = torch.softmax(logits, dim=-1) 
            next_token = torch.multinomial(probs, num_samples=1)
            print(f"next token is {next_token}")
            # print(f"next toke is {self.tokenizer.decode(next_token.item(), skip_special_tokens=True)}")
            generated_tokens.append(next_token.item())
            # print(f"eos is {self.tokenizer.eos_token_id}")
            if next_token.item() == self.tokenizer.eos_token_id:
                print("eos")
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        return output_text
    """
    
    @classmethod
    def from_pretrained(cls, modelpath, tokenizerpath):
        sd = load_file(modelpath)
        original_model_path = "/vepfs-mlp/mlp-public/zihaoyang/LLaMA/tinyllama"
        LLM = AutoModelForCausalLM.from_pretrained(original_model_path)
        LLM.resize_token_embeddings(32001)
        tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
        
        model = Augmented_LLM(LLM, tokenizer, config.n_regs, config.n_val, config.prog_max_length, config.dropout, LLM.config.hidden_size, config.max_tokens, config.intermidate_dim)
        _,_ = model.load_state_dict(sd, strict=False)
        return model
