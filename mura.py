import torch
import math
from torch import nn
from torch.autograd import Function

@torch.compile
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class LinearFunction(Function):
    """
    y = x @ (W ^T + UV)+ b
    Shapes:
      x: (..., in_features)
      W: (out_features, in_features)
      b: (out_features,) or None
      y: (..., out_features)
    """
    @staticmethod
    def forward(ctx, x, weight, u, v, bias=None):
        # Save tensors for backward (keep as few as you need)
        ctx.save_for_backward(x, weight, u, v, bias if bias is not None else torch.tensor([]))
        # Also save sizes to support ND batch shapes without extra views in backward
        ctx.in_features = weight.size(1)
        ctx.out_features = weight.size(0)
        # Forward: y = x @ W^T + b
        y = x.matmul(weight.t() + u @ v)
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def backward(ctx, grad_out):
        """
        Given dL/dy (= grad_out), return tuple of gradients
        (dL/dx, dL/dW, dL/dbias). Return None for non-tensor inputs.
        """
        x, weight, u, v, bias = ctx.saved_tensors
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # grad_out: (..., out_features)

        # Compute gradients:
        # dL/dx = dL/dy @ W
        grad_x = grad_out.matmul(weight.to(grad_out.dtype))

        # dL/dW = (dL/dy)^T @ x  with batch dims collapsed together.
        # Collapse leading dims (if any) into one big batch for the matmul
        # so we can write it as: (B, out)T @ (B, in) = (out, in)
        if x.ndim == 2:
            B = x.shape[0]
            x2 = x
            go2 = grad_out
        else:
            B = x.numel() // x.shape[-1]
            x2 = x.reshape(B, x.shape[-1])
            go2 = grad_out.reshape(B, grad_out.shape[-1])
        grad_weight = go2.t().matmul(x2.to(go2.dtype))
        
        # normalize grad weight with newton schulz
        ns_grad_weight = zeropower_via_newtonschulz5(grad_weight, steps = 5).to(v.dtype)
        grad_u = (v @ ns_grad_weight).T
        grad_v = (ns_grad_weight @ u).T

        # dL/db = sum over batch of dL/dy
        grad_bias = None
        if bias.numel() != 0:
            # sum over all batch dimensions
            reduce_dims = list(range(grad_out.ndim - 1))
            grad_bias = grad_out.sum(dim=reduce_dims)

        return grad_x, grad_weight, grad_u, grad_v, grad_bias


class Linear(nn.Module):
    """
    Module wrapper around LinearFunction with explicit params, like nn.Linear.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, k = 64):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.u = nn.Parameter(torch.zeros((in_features, k), **factory_kwargs))
        self.v = nn.Parameter(torch.zeros((k, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Same spirit as nn.Linear default init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, +bound)

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.u, self.v, self.bias)


# --------- Quick correctness check (compare to nn.Linear) ----------
if __name__ == "__main__":
    import math
    torch.manual_seed(0)
    N, In, Out = 4, 3, 5
    x = torch.randn(N, In, requires_grad=True)

    ref = nn.Linear(In, Out)
    mine = Linear(In, Out)
    # copy params so they match
    with torch.no_grad():
        mine.weight.copy_(ref.weight)
        if mine.bias is not None:
            mine.bias.copy_(ref.bias)

    y_ref = ref(x)
    y_mine = mine(x)
    assert torch.allclose(y_ref, y_mine, atol=1e-6), "forward mismatch"

    loss_ref = y_ref.pow(2).mean()
    loss_mine = y_mine.pow(2).mean()

    loss_ref.backward()
    loss_mine.backward()
    assert torch.allclose(x.grad, x.grad.clone(), atol=1e-6)  # sanity for autograd state
    assert torch.allclose(ref.weight.grad, mine.weight.grad, atol=1e-6), "dW mismatch"
    if ref.bias is not None:
        assert torch.allclose(ref.bias.grad, mine.bias.grad, atol=1e-6), "db mismatch"
