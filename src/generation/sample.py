"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from src.gpt2.gpt2_model import GPTConfig, GPT
from src.common.utils.compute_utils import get_device_type, get_pt_dtype, get_ctx

model_args = {}

def setup(
    init_from: str ='gpt2',  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir: str = 'out',  # ignored if init_from is not 'resume'
    start: str = "\n",  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 10,  # number of samples to draw
    max_new_tokens: int = 500,  # number of tokens generated in each sample
    temperature: float = 0.8,  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 200,  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337,
    device: str = 'cuda',  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = True,  # use PyTorch 2.0 to compile the model to be faster
):
    global model_args
    model_args = {"out_dir"                    : out_dir,
                  "init_from"                  : init_from,
                  "device"                     : device,
                  "dtype"                      : dtype,
                  "compile"                    : compile,
                  "seed"                       : seed,
                  "device_type"                : get_device_type(device),
                  "pt_dtype"                   : get_pt_dtype(dtype)
    }
    # for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # not sure if it is needed at all. torch.manual_seed() should be enough.

    # allow tf32 on matmul on new NVIDIA GPUs since Ampere. Defaults to False for pytorch 1.12 and later.
    # tf32 stands for TensorFloat32 tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul.
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    ctx = get_ctx(model_args["device"], model_args["dtype"])

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        print("overriding dropout rate to 0.")
        gptconf = GPTConfig(
            block_size = model_args["block_size"],
            vocab_size = model_args["vocab_size"],
            n_layer = model_args["n_layer"],
            n_head = model_args["n_head"],
            n_embd = model_args["n_embd"],
            dropout = 0.,
            bias = model_args["bias"]
        )
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    print("Setting model to evaluation mode..")
    model.eval()

    print(f"Moving model to device: {device}..")
    model.to(device)

    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'model_args' in checkpoint and 'dataset' in checkpoint['model_args']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['model_args']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    # 1d tensor
    start_ids = encode(start)
    # 2d tensor: [1, n]
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)