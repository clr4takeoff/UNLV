from torchvision.datasets import CIFAR10 as _CIFAR10
from typing import Union, Optional, Callable
from pathlib import Path

from .base import BaseDataset


class CIFAR10(BaseDataset, _CIFAR10):
    dataset_name = "cifar-10"
    num_classes = 10

    img_size = 32
    img_mean = (0.4914, 0.4822, 0.4465)
    img_std = (0.2470, 0.2435, 0.2616)

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(Path(root)/self.dataset_name, train, transform, target_transform, download)
