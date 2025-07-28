from differentiable_computer.config import program
import torch
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
        print(f"grad output is {grad_output}")
        input, = ctx.saved_tensors  
        grad_input = grad_output.unsqueeze(-1).expand_as(input)  # (B, L, 1) -> (B, L, 16)
        print(f"grad input is {grad_input.shape}")
        return grad_input

class Programable(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, onehot):
        idx       = {k: oh.argmax(-1) for k, oh in onehot.items()} 
        print(f"idx is {idx}")
        program_assemble = ((idx["op"]  << 12) | (idx["dst"] <<  9) | (idx["src1"]<<  6) | (idx["src2"]<<  3) | idx["imm"]) # (B, L, max_length)
        return program_assemble
    
    @staticmethod
    def backward(ctx, grad_output):
        print(f"Backward for choosing index")
        print(f"grad output is {grad_output}")
        return grad_output
    
if __name__ == '__main__':
    converter = Convert.apply
    x = torch.tensor([[[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9],[0.1,0.1,0.2,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]]], requires_grad=True)
    output = converter(x)
    print(output)
    
    loss = ((1.0-output)**2).mean()
    loss.backward()
    print(x.grad.shape)
    