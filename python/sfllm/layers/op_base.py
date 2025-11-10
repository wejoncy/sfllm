from torch import nn

class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def dispatch_forward(self):
        return self.forward_native


    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
