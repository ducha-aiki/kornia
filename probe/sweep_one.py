"""One depthwise conv2d: sweep_one.py DTYPE C H OW K ORIENT(h|v) MKLDNN(on|off)."""

import sys

import torch
import torch.nn.functional as F

dtype_name, c, h, ow, k, orient, mkldnn = sys.argv[1:8]
c, h, ow, k = int(c), int(h), int(ow), int(k)
dtype = getattr(torch, dtype_name)
if orient == "h":
    x, w = torch.rand(1, c, h, ow + k - 1, dtype=dtype), torch.rand(c, 1, 1, k, dtype=dtype)
else:
    x, w = torch.rand(1, c, ow + k - 1, h, dtype=dtype), torch.rand(c, 1, k, 1, dtype=dtype)
with torch.backends.mkldnn.flags(enabled=mkldnn == "on"):
    F.conv2d(x, w, groups=c)
print("ok", flush=True)
