
gpu_promised_tflops_map = {"A100": 312e12, "4090": 165e12}  # GPU bfloat16 peak flops
my_gpu_name: str = "4090"

# DDP settings
backend: str = 'nccl'  # 'nccl', 'gloo', etc.