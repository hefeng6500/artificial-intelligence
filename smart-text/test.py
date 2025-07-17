import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # 输出True表示Metal加速可用