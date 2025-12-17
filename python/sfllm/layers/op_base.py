from torch import nn
import torch

class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def dispatch_forward(self):
        if hasattr(self, 'forward_cuda') and torch.cuda.is_available():
            return self.forward_cuda
        return self.forward_native


    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
    
    # Dictionary of all custom ops (classes, indexed by registered name).
    # To check if an op with a name is enabled, call .enabled() on the class.
    # Examples:
    # - MyOp.enabled()
    # - op_registry["my_op"].enabled()
    op_registry: dict[str, type["CustomOp"]] = {}
    op_registry_oot: dict[str, type["CustomOp"]] = {}

    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator
