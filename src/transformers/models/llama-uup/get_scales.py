from math import e, erf, pi, sqrt, exp
from typing import *

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from unit_scaling.constraints import apply_constraint
from unit_scaling.scale import scale_fwd, scale_bwd

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
def check_scaling(fn: Callable[[Tensor], Tensor], **kwargs: Any) -> None:
    x = torch.randn(int(1e6)).requires_grad_()
    y = fn(x, **kwargs)
    y.backward(torch.randn_like(y))

    name = f"{fn.__module__}.{fn.__name__}".replace("__main__.", "")
    print(name + (f" {kwargs}" if kwargs else ""))
    for k, v in {"x": x, "y": y, "grad_x": x.grad}.items():
        print(f"{k:>10}.std = {v.std(correction=0).item():.3f}")

check_scaling()