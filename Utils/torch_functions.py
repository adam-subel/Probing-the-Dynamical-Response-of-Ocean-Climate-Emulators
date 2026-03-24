import torch

class CappedGELU(torch.nn.Module):

    def __init__(self, cap_value=1.0):
        super().__init__()
        self.gelu = torch.nn.GELU()
        # self.cap = torch.tensor(cap_value, dtype=torch.float32)
        self.cap =  cap_value

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x