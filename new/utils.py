import torch

def compute_acc(labels,predict):
    """
    labels (b,)
    predict (b,n)
    """
    pred = predict.argmax(dim=1)
    correct = (pred == labels).sum().item()
    
    acc = correct.float() / labels.size(0)
    
    return acc

def count_correct(labels,predict):

    pred = predict.argmax(dim=1)
    correct = (pred == labels).sum().item()
    
    return correct

def count_correct_multibits(labels, predict):
    """
    predict (b,bits,10)
    labels (b,bits)
    """
    pred = predict.argmax(dim=1) #(b,bits)
    correct = (pred == labels)
    correct_all = correct.all(dim=1)
    
    num = correct_all.sum().item()
    return num



def grad_norm(parameters, norm_type=2, except_gate=True):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if except_gate:
        params = [x for x in parameters if ('Gate' not in x[0] and x[1].grad is not None)]
    else:
        params = [x for x in parameters if x[1].grad is not None]
    if len(params) == 0:
        return 0.0
    if norm_type == float('inf'):
        total = max(p.grad.detach().abs().max().item() for p in params)
        return float(total)
    total = 0.0
    total_num = 0
    for name, p in params:
        total_num += p.numel()
        param_norm = p.grad.detach().float().norm(norm_type)
        total += (param_norm.item() ** norm_type)
    
    total_norm = float(total ** (1.0 / norm_type))
    rms = total_norm / (total_num ** 0.5 + 1e-6)
    return total_norm, rms