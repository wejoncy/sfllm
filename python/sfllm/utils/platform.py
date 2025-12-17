from functools import lru_cache
import torch

class CUDAPlatformUtils:
    """Utility functions for CUDA platform."""

    def has_device_capability(
        self,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """Check if the CUDA device has the specified capability.

        Args:
            capability (tuple[int, int] | int): The capability to check. Can be a
                tuple (major, minor) or an integer representing major version.
            device_id (int, optional): The CUDA device ID. Defaults to 0.
        Returns:
            bool: True if the device has the specified capability, False otherwise.
        """
        if isinstance(capability, int):
            major = capability
            minor = 0
        else:
            major, minor = capability

        device_capability = torch.cuda.get_device_capability(device_id)
        return (device_capability[0] > major) or (
            device_capability[0] == major and device_capability[1] >= minor
        )

    def get_device_name(self, device_id: int = 0) -> str:
        """Get the name of the CUDA device.

        Args:
            device_id (int, optional): The CUDA device ID. Defaults to 0.
        Returns:
            str: The name of the CUDA device.
        """
        return torch.cuda.get_device_name(device_id)
    
    def get_device_core_count(self, device_id: int = 0):
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_properties(device_id).multi_processor_count
        return 0

    # https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
    @lru_cache(maxsize=1)
    def is_hip(self) -> bool:
        return torch.version.hip is not None

    @lru_cache(maxsize=1)
    def is_cuda(self):
        return torch.cuda.is_available() and torch.version.cuda

    @lru_cache(maxsize=1)
    def is_cuda_alike(self):
        return self.is_cuda() or self.is_hip()

    @lru_cache(maxsize=1)
    def is_blackwell(self) -> bool:
        if not self.is_cuda():
            return False
        return torch.cuda.get_device_capability()[0] == 10


current_platform = CUDAPlatformUtils()