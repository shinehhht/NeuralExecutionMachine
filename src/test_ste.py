import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Convert(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        b = (input>=0.5).int()
        weights = 2 ** torch.arange(3,-1,-1, device=b.device)
        unsigned_val = (b * weights).sum(-1)
        signed_val = torch.where(unsigned_val >= 2**3, unsigned_val - 2**4, unsigned_val)
        # print(f"signed value is {signed_val.float()}")
        return signed_val.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # print(f"Backward for converting registers value")
        # print(f"grad inputput is {grad_output}")
        #print(f"hhhhgrad input shape is {grad_output.shape}")
        input, = ctx.saved_tensors  
        grad_input = grad_output.expand_as(input)  # (B, L, 1) -> (B, L, 4)
        #print(f"grad output is {grad_input}")
        return grad_input

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class transform15(nn.Module):
    def __init__(self, r_dim, n_regs, n_val):
        super().__init__()
        self.n_regs = n_regs
        self.n_val = n_val
        self.w = nn.Linear(r_dim, n_regs * n_val)
        self.converter = Convert.apply
        
    def forward(self, z_hidden):
        print(f"\n\nParameter is {self.w.weight}\n\n")
        proj = torch.sigmoid(self.w(z_hidden)).view(1, self.n_regs, self.n_val)
        print(f"proj is {proj}")
        bitform = (proj >= 0.5).float()
        print(f"bit is {bitform}")
        R_head = torch.sigmoid(self.w(z_hidden)).view(1, self.n_regs, self.n_val)
        #print(f"R_head shape is {R_head.shape}")
        R = self.converter(R_head) # (1,1)
        return bitform, R, R_head
    
set_seed(42)   
model = transform15(r_dim=4, n_regs=1, n_val=4).to("cuda")
"""
with torch.no_grad():
    model.w.weight.fill_(1.0)
    model.w.bias.fill_(0.0)
"""
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

z_hidden = torch.randn(1,4).cuda()
# z_hidden = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device='cuda')
print(f"z_hidden is {z_hidden}")
for step in range(4):
    print(f"\n\nstep {step}\n\n")
    bitform, output_value,origin_value = model(z_hidden)
    print(f"R_head is {origin_value}, bitform is {bitform.shape}")
    
    target_value_bit = torch.tensor([[[0.,1.,1.,0.]]], requires_grad=True).to("cuda")
    # target_value = torch.full_like(output_value, 6.0)
    # print(f"target shape is {target_value.shape}")
    loss = F.mse_loss(bitform, target_value_bit)
    print(f"loss is {loss}")

    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:\n{param.grad}")
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Output = {output_value.item():.2f}, Loss = {loss.item():.4f}")
        print("Binary bits:", (origin_value >= 0.5).int().squeeze().tolist())

        