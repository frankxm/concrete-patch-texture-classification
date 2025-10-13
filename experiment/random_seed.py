import torch
import numpy as np
import random
# torch.manual_seed(seed) 设定 PyTorch 的 CPU 计算的随机种子，确保 CPU 端的计算是可复现的。任何涉及pytorch的随即操作
# torch.cuda.manual_seed_all(seed)设定 PyTorch 在 GPU（CUDA）上的随机种子，确保 GPU 端的计算结果一致。这里使用 manual_seed_all(seed)，是因为如果有多个 GPU，它们都需要相同的随机种子。
#  np.random.seed(seed)设定 NumPy 的随机种子，保证 NumPy 生成的随机数是可复现的。
# torch.backends.cudnn.deterministic = True 让 cuDNN（NVIDIA 的 GPU 计算库）使用确定性算法，而不是默认的非确定性优化算法（有些 cuDNN 操作可能具有非确定性行为）。副作用：可能会降低训练速度，因为某些 cuDNN 计算模式是非确定性的，但更快。
def setup_seed(seed):
    # torch.initial_seed() 在 seed_worker 里不是完全随机的，它是根据主进程的种子推导出来的，只要主进程的种子（torch.manual_seed()）是固定的，torch.initial_seed() 的结果也是确定的。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 可能会降低训练速度



