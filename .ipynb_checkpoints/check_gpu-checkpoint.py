import torch

if torch.cuda.is_available():
    print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")