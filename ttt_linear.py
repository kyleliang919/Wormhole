import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init="xavier"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale_weight = nn.Parameter(torch.empty(1, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        self.reset_parameters(init)
        self.prev_x = None
        self.error = None
        self.time_dim = 1

    def reset_parameters(self, init):
        if init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.scale_weight)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(self.scale_weight, nonlinearity='linear')
        else:
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.scale_weight, std=0.02)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            if self.prev_x is None:
                out = F.linear(x, self.weight, self.bias)
                if x.shape[self.time_dim] == 1:
                    self.prev_x = torch.concat([self.prev_x, x], dim = self.time_dim)
                else:
                    self.prev_x = x  
            elif self.error is None:
                out = F.linear(x, self.weight, self.bias)
                _, self.error = torch.chunk(out, 2, dim = self.time_dim)
            else:
                error_ = self.error.reshape(-1, self.out_features)
                prev_x_ = self.prev_x.reshape(-1, self.in_features)
                scale = F.linear(prev_x_, self.scale_weight)
                breakpoint()
                delta_weight = error_.T @ (prev_x_ *  scale)
                
                print(delta_weight)
                out = F.linear(x, self.weight + delta_weight, self.bias)
                self.prev_x = None
                self.error = None
        else:
            out = F.linear(x, self.weight, self.bias)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

if __name__=="__main__":
    model = Linear(64, 64).cuda()
    x = torch.randn(2, 128, 64).cuda()
    y = torch.randn(2, 128, 64).cuda()
    out1 = model(x)
    out2 = model(torch.cat([x, y], dim = 1))
    out3 = model(x)
    out3.sum().backward()
    ttt_grad = model.weight.grad.clone()
    print(model.weight.grad)
    print(out3)

    