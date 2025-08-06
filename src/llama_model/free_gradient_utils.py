import torch
import torch.nn.functional as F
import random


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

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
class sampleTool():
    """
    
    logits = {
        "op":(B, L, self.prog_max_length, 4),
        "dst": (B, L, self.prog_max_length, 3),
        "src1": (B, L, self.prog_max_length, 3),
        "src2": (B, L, self.prog_max_length, 3),
        "imm" : (B, L, self.prog_max_length, 3)
    }
    
    R (B,L,n_regs,8)
    prog_len (B,L,4)
        
    """
    def __init__(self,config):
        super().__init__()
        self.max_proglen = config.prog_max_length
        self.op_bit = config.n_op
        self.dst_bit = config.n_dst
        self.src1_bit = config.n_src1
        self.src2_bit = config.n_src2
        self.imm_bit = config.n_imm
        self.n_regs = config.n_regs
        self.r_bit = config.n_val
        self.prog_len_bit = config.pc_bit
    
    def bitflip(self, tensor, flip_num, seed):
        if seed:
            torch.manual_seed(seed)
        
        origin_shape = tensor.shape
        flat = tensor.reshape(-1, origin_shape[-1])
        B,D = flat.shape
        flipped = flat.clone()
        all_perm = torch.rand(D * B, device=tensor.device).view(B, D)
        flip_num = min(flip_num, D)  
        rand_idx = all_perm.argsort(dim=1)[:, :flip_num]
        batch_idx = torch.arange(B, device=tensor.device).unsqueeze(1).expand(-1, flip_num)

        flipped[batch_idx, rand_idx] = 1 - flipped[batch_idx, rand_idx]

        return flipped.view(origin_shape)
    
    def aggregation(self,R,logits,prog_len):
        logits_list = [logits[k] for k in ['op','dst','src1','src2','imm']]
        prog_bits = torch.cat(logits_list, dim=-1) # (B,L,prog_len,16)
        B,L = prog_bits.size(0),prog_bits.size(1)
        
        prog_bits_flat = prog_bits.view(B, L, -1) # (B,L,256)
        R_flat = R.view(B,L,-1) # (B,L,8*16)
        origin_code = torch.cat([prog_bits_flat, R_flat, prog_len], dim=-1) # (B,L,388)
        
        return origin_code
    
    def disgregation(self, tensor, table=None):
        out = {}
        logits = {}
        B = tensor.shape[0]
        L = tensor.shape[1]
        if not table:
            table = self.build_table()
            
        for key, (start, end) in table.items():
            field_tensor = tensor[:, :, start:end] 
            if key == 'op':
                logits[key] = field_tensor.view(B, L, self.max_proglen, self.op_bit)
            elif key == 'dst':
                logits[key] = field_tensor.view(B, L, self.max_proglen, self.dst_bit)
            elif key == 'src1':
                logits[key] = field_tensor.view(B, L, self.max_proglen, self.src1_bit)
            elif key == 'src2':
                logits[key] = field_tensor.view(B, L, self.max_proglen, self.src2_bit)
            elif key == 'imm':
                logits[key] = field_tensor.view(B, L, self.max_proglen, self.imm_bit)
            elif key == 'R':
                out[key] = field_tensor.view(B, L, self.n_regs, self.r_bit)
            elif key == 'prog_len':
                out[key] = field_tensor  
        out['logits'] = logits
        
        return out
        
    def build_table(self):
        field_bit_lengths = {
            "op": self.max_proglen * self.op_bit,       
            "dst": self.max_proglen * self.dst_bit,
            "src1": self.max_proglen * self.src1_bit,
            "src2": self.max_proglen * self.src2_bit,
            "imm": self.max_proglen * self.imm_bit,
            "R": self.n_regs * self.r_bit,         
            "prog_len": self.prog_len_bit,
        }
        
        table = {}
        offset = 0
        for name, size in field_bit_lengths.items():
            table[name] = (offset, offset + size)
            offset += size
            
        return table
    
    def num_to_flip(self,max_flip_num,seed):
        if seed is not None:
            random.seed(seed)
        flip_num = random.randint(1, max_flip_num) 
        return flip_num
    
    def generate(self, Plogits, R, prog_len, group):
        pass
        
        

class GroupSample(sampleTool):
    
    def generate(self, current_set, group, flip_num):
        
        R = current_set['R']
        Plogits = current_set['logits']
        prog_len = current_set['prog_len']
       
        R_origin = (R >= 0.5).int()
        logits_origin = {k: (v >=0.5).int() for k, v in Plogits.items()}
        prolen_origin = (prog_len >= 0.5).int()
        
        current_set = {
            'R':R_origin,
            'logits':logits_origin,
            'prog_len':prolen_origin
        }
        
        # print(f"R_origin is {R_origin.shape}")
        candidate_set = []
        candidate_set.append(current_set)
        for i in range(group):
            # print(f"group {i}")
            R_flip = self.bitflip(R_origin, flip_num, seed=i)
            # print(f"R_flip is {R_flip.shape}")
            logits_flip = {
                k:self.bitflip(v,flip_num,seed=i+100+k_i) for k_i,(k,v) in enumerate(logits_origin.items())
            }
            prog_len_flip = self.bitflip(prolen_origin, flip_num, seed=i+200)

            candidate_set.append({
                "R":R_flip,
                "logits":logits_flip,
                "prog_len":prog_len_flip
            })
        
        return candidate_set
    
    def generate4R(self, current_set, group, flip_num):
        R = current_set['R']
        Plogits = current_set['logits']
        prog_len = current_set['prog_len']
       
        R_origin = (R >= 0.5).int()
        logits_origin = {k: (v >=0.5).int() for k, v in Plogits.items()}
        prolen_origin = (prog_len >= 0.5).int()
        
        current_set = {
            'R':R_origin,
            'logits':logits_origin,
            'prog_len':prolen_origin
        }
        
        # print(f"R_origin is {R_origin.shape}")
        candidate_set = []
        candidate_set.append(current_set)
        for i in range(group):
            R_flip = self.bitflip(R_origin, flip_num, seed=i)

            candidate_set.append({
                "R":R_flip,
                "logits":logits_origin,
                "prog_len":prolen_origin
            })
        
        return candidate_set


class FieldsSample(sampleTool):

    def generate(self, current_set, group, portion, max_fields_flip=5):

        R = current_set['R']
        Plogits = current_set['logits']
        prog_len = current_set['prog_len']
       
        R_origin = (R >= 0.5).int() 
        logits_origin = {k: (v >=0.5).int() for k, v in Plogits.items()}
        prolen_origin = (prog_len >= 0.5).int()
        
        current_set = {
            'R':R_origin,
            'logits':logits_origin,
            'prog_len':prolen_origin
        }
        
        
        candidate_set = []
        candidate_set.append(current_set)
        
        table = self.build_table()
        field_names = list(table.keys())
        origin_code = self.aggregation(R_origin,logits_origin,prolen_origin)
        
        filtered_items = {k: v for k, v in portion.items() if v >= 0.005}
        filtered_keys = list(filtered_items.keys())
        choice_weight = torch.tensor(list(filtered_items.values()))
        min_flip = 2
        max_ratio = 0.5
        scaling_factor = 0.2
        
        for i in range(group):

            num_fields = random.randint(1, min(max_fields_flip+1, len(filtered_keys)))
            
            indices = torch.multinomial(choice_weight, num_samples=num_fields, replacement=False)
            #print(f"indices is {indices}")
            chosen_fields = [field_names[i] for i in indices]
            new_code = origin_code.clone()
            
            
            for index, f in enumerate(chosen_fields):
                # print(f"flip {f}")
                start, end = table[f]
                tensor = new_code[:,:,start:end] # (B,L, )
                flip_num = int(min_flip + tensor.shape[-1] * (scaling_factor * portion[f] + random.uniform(0.02, 0.05)))
                flip_num = min(flip_num, int(tensor.shape[-1] * max_ratio))
                # print(f"flip num is {flip_num}")
                new_code[:,:,start:end] = self.bitflip(tensor, flip_num=flip_num,seed=i*index)
                # print(f"after flip {new_code[0,0,start:end]}")
            
            candidate_set.append(self.disgregation(new_code,table))
        
        # print(f"candidate_set length is {len(candidate_set)}")
        return candidate_set
              
        
if __name__ == '__main__':
    # set_seed(42) 
    # tool = groupSample(4,3,3,3,3,8,4)
    tool = FieldsSample(16,4,3,3,3,3,8,16,4)
    B = 2              
    L = 3                
    prog_max_length = 16
    n_regs = 8
    

    logits = {
        "op":   torch.sigmoid(torch.randn(B, L, prog_max_length, 4)),
        "dst":  torch.sigmoid(torch.randn(B, L, prog_max_length, 3)),
        "src1": torch.sigmoid(torch.randn(B, L, prog_max_length, 3)),
        "src2": torch.sigmoid(torch.randn(B, L, prog_max_length, 3)),
        "imm":  torch.sigmoid(torch.randn(B, L, prog_max_length, 3))
    }

    R = torch.sigmoid(torch.randn(B, L, n_regs, 16))
    prog_len = torch.sigmoid(torch.rand(B, L, 4))

    current_set = {
        'R':R,
        'logits':logits,
        'prog_len':prog_len
    }
    
    portion = {
        'op':0.1,
        'dst':0.1,
        'src1':0.1,
        'src2':0.1,
        'imm':0.1,
        'R':0.1,
        'proglen':0.1
    }
    candidate=tool.generate(current_set,group=1,portion=portion,max_fields_flip=5)
    
    """
    R_bit = torch.stack([c['R'] for c in candidate], dim=2) 
    logits_bit = { k: torch.stack([c['logits'][k] for c in candidate], dim=2) for k in candidate[0]['logits'].keys()}
    print(logits_bit)
    print(f"shape is {logits_bit['op'].shape}")
    P = program_bit2decimal(logits_bit)
    print(P)
    print(P.shape)
    """
    
        