from torchvision.datasets import CIFAR100 as _CIFAR100
from typing import Union, Optional, Callable
from pathlib import Path

from .base import BaseDataset


class CIFAR100(BaseDataset, _CIFAR100):
    dataset_name = "cifar-100"
    num_classes = 100

    img_size = 32
    img_mean = (0.5071, 0.4865, 0.4409)
    img_std = (0.2673, 0.2564, 0.2762)

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(Path(root)/self.dataset_name, train, transform, target_transform, download)
