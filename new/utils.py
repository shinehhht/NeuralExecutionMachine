import torch
import torch.nn as nn
import torch.nn.functional as F
     

class Convert(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        b = (input>=0.5).int()
        weights = 2 ** torch.arange(15,-1,-1, device=b.device)
        unsigned_val = (b * weights).sum(-1)
        signed_val = torch.where(unsigned_val >= 2**15, unsigned_val - 2**16, unsigned_val)
        # print(f"signed value is {signed_val.float()}")
        return signed_val.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        print(f"Backward for converting registers value")
        print(f"grad inputput is {grad_output}")
        print(f"grad input shape is {grad_output.shape}")
        input, = ctx.saved_tensors  
        grad_input = grad_output.unsqueeze(-1).expand_as(input)  # (B, L, 1) -> (B, L, 16)
        print(f"grad output is {grad_input.shape}")
        return grad_input


def R_bit2decimal(tensor):
    """
    R_bit  (B, L, group_num, n_regs, 16)
    """
    width = tensor.shape[-1]
    weights = 2 ** torch.arange(width-1,-1,-1, device=tensor.device)
    unsigned_val = (tensor * weights).sum(-1)
    signed_val = torch.where(unsigned_val >= 2**(width-1), unsigned_val - 2**width, unsigned_val)
    return signed_val.float() # (B,L,group_num,n_regs)

def bit2decimal(tensor):
    width = tensor.shape[-1]
    weights = 2 ** torch.arange(width-1, -1, -1, device=tensor.device)
    return (tensor * weights).sum(-1), width

def program_bit2decimal(logits):
    """
    logits = {
        "op":   (B, L, prog_max_length, 4),
        "dst":  (B, L, prog_max_length, 3),
        "src1": (B, L, prog_max_length, 3),
        "src2": (B, L, prog_max_length, 3),
        "imm":  (B, L, prog_max_length, 3)
    }
    
    return (B, L,prog_max_length)
    """
    op, _ = bit2decimal(logits["op"])   
    
    dst, dst_bit = bit2decimal(logits["dst"])  
    src1, src1_bit = bit2decimal(logits["src1"]) 
    src2, src2_bit = bit2decimal(logits["src2"])
    imm, imm_bit = bit2decimal(logits["imm"])  

    
    # 拼接为 16-bit 指令
    instruction = (
        (op   <<  imm_bit+src2_bit+src1_bit+dst_bit) |  
        (dst  <<  imm_bit+src2_bit+src1_bit) |
        (src1 <<  imm_bit+src2_bit) |
        (src2 <<  imm_bit) |
        imm
    ) 

    return instruction
    

def assistLoss(current_set, best_candidate):
    """
    logits = {
            "op":(B, L, self.prog_max_length, 4),
            "dst": (B, L, self.prog_max_length, 3),
            "src1": (B, L, self.prog_max_length, 3),
            "src2": (B, L, self.prog_max_length, 3),
            "imm" : (B, L, self.prog_max_length, 3)
        }
        
    R (B,L,n_regs,16)
    prog_len (B,L,4)
        
    candidate = {
        "R": R_flip,
        "logits": logits_flip,
        "prog_len": prog_len_flip   
    }
    
    """
    total_loss = 0.0
    field_loss = {}
    
    pc_width = best_candidate['prog_len'].shape[-1]
    weights = 2 ** torch.arange(pc_width-1, -1, -1, device=best_candidate['prog_len'].device)
    valid_length = (best_candidate['prog_len'] * weights).sum(dim=-1) # (B,L)
    loss_fn = nn.BCELoss(reduction='none')
    # print(f"curretn R {current_set['R'].shape}")
    # print(f"gold R shape {best_candidate['R'].shape}")
    for key in current_set:
        if key == 'logits':
            for sub_key in current_set[key]:
                cur = current_set[key][sub_key]
                # print(f"cur shape is {cur.shape}")
                target = best_candidate[key][sub_key].float()
                
                if target.shape != cur.shape:
                    # print(f"key {sub_key}, target shape {target.shape}, cur shape is {cur.shape}")
                    target = target.expand_as(cur)
                    
                B,L,P,D = cur.shape
                step_ids = torch.arange(P, device=cur.device).view(1, 1, -1)
                valid_len_expand = valid_length.unsqueeze(-1)  # (B, L, 1)
                mask = (step_ids < valid_len_expand).unsqueeze(-1)  # (B, L, P, 1)
            
                loss = loss_fn(cur,target) * mask
                num_valid = mask.expand_as(loss).sum()

                reduced_loss = loss.sum() # / mask.sum().clamp(min=1)  
                field_loss[sub_key] = reduced_loss / num_valid.clamp(min=1) 
                total_loss += reduced_loss
                
        else:
            cur = current_set[key]
            target = best_candidate[key].float()
            if target.shape != cur.shape:
                target = target.expand_as(cur)
            loss = loss_fn(cur,target) 
            reduced_loss = loss.sum()

            if key =='R':
                field_loss[key] = loss.mean()
            else:
                field_loss[key] = loss.mean()
                
            total_loss += reduced_loss
         
    # portion_dict = {k: (v / total_loss).item() for k, v in field_loss.items()}
    
    return total_loss, field_loss


def assistLoss4R(current_set, best_candidate):
    loss_fn = nn.BCELoss(reduction='none')
    field_loss = {}
    for key in current_set:
        if key == 'R':
            cur = current_set[key]
            target = best_candidate[key].float()
            total_loss = loss_fn(cur,target)
            reduced_loss = total_loss.sum()
            field_loss[key] = total_loss.mean()
            
    return reduced_loss, field_loss

def computportions(percentage_list,temperature=1.0, normalized=False):
    total = {}
    for d in percentage_list:
        for k, v in d.items():
            if k not in total:
                total[k] = v
            else:
                total[k] = total[k] + v 
    
    keys = list(total.keys())
    values = torch.tensor([total[k] for k in keys], dtype=torch.float32)
    
    """
    total_sum = sum(total.values()) + 1e-6
    normalized = {k:v / total_sum for k,v in total.items()}
    """
    
    if normalized:
        values = (values - values.min()) / (values.max() - values.min() + 1e-6)
        
    soft_values = F.softmax(values / temperature, dim=0)  

    normalized = {k: soft_values[i] for i, k in enumerate(keys)}
    
    # print(f"percen list {len(percentage_list)}")
    total = {k:v/len(percentage_list) for k,v in total.items()}
    
    return total, normalized

def update_next_portions(alpha, cur_portions, prev_portions):
    return  {
        key: alpha * cur_portions.get(key, 0.0) + (1 - alpha) * prev_portions.get(key, 0.0)
        for key in set(cur_portions) | set(prev_portions)
    }

"""
def update_grad(alpha, cur_portions, prev_portions):
"""
  
def hook_grads(current_set):
    saved_grads = {}
            
    def save_grad(name):
        def hook(grad):
            saved_grads[name] = grad
        return hook
    
    current_set['R'].register_hook(save_grad('R'))
    current_set['prog_len'].register_hook(save_grad('prog_len'))
    for k,v in current_set['logits'].items():
        v.register_hook(save_grad(k))
        
    return saved_grads


if __name__ == '__main__':
    converter = Convert.apply
    #x = torch.tensor([[[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9],[0.1,0.1,0.2,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]]], requires_grad=True)
    x = torch.tensor([[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.6,0.5,0.9,0.1]]], requires_grad=True)
    output = converter(x)
    print(output)
    
    loss = ((15-output)**2).mean()
    loss.backward()
    print(x.grad.shape)
    