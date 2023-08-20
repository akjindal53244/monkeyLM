import torch
from contextlib import nullcontext

def get_device_type(device):
    return 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast

def get_tokens_per_iter(block_size, micro_batch_size, gradient_accumulation_steps) -> int:
        return block_size * micro_batch_size * gradient_accumulation_steps

def get_pt_dtype(dtype):
    # note: float16 data type will automatically use a GradScaler
    return {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

def get_ctx(device, dtype):
    device_type = get_device_type(device)
    pt_dtype = get_pt_dtype(dtype)
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = pt_dtype)
    return ctx

