from torch.utils.data import Dataset


class BaseDataset(Dataset):
    dataset_name: str = "dataset"
    num_classes: int = 0
    img_size: int = 0
    image_mean: list[float] = (0, 0, 0)
    image_std: list[float] = (1, 1, 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.transform is None:
            self.transform = transforms.ToTensor()
