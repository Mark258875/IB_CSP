import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version (compiled):", torch.version.cuda)
    print("Device 0:", torch.cuda.get_device_name(0))