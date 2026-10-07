import torch
import torch.nn.functional as F

x = torch.rand(1, 3, 96, 112, dtype=torch.float16)
w = torch.rand(3, 1, 1, 17, dtype=torch.float16)
print(torch.__version__, "start", flush=True)
y = F.conv2d(x, w, groups=3)  # never returns on Xeon 6973P-C / Xeon Platinum 8573C
print("done", tuple(y.shape), flush=True)
