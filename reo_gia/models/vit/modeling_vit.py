from transformers import ViTForImageClassification as _ViTForImageClassification, ViTConfig
from ..base import BaseModel, ModelRegistry


class ViTForImageClassification(BaseModel, _ViTForImageClassification):
    model_id = "WinKawaks/vit-small-patch16-224"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
    })


ViTSmall = ViTForImageClassification  # Alias


class ViTForImageClassification(BaseModel, _ViTForImageClassification):
    model_id = "google/vit-base-patch16-224"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': "n1kooo/vit-cifar10"
    })


ViTBase = ViTForImageClassification  # Alias