# my_project/my_project/models/factory.py

from .custom_cnn import CNN5d
from .pretrained import (
    get_resnet18,
    get_resnet50,
    get_vit_b16  # <-- 記得 import
)
from ..config import Config

def get_model(model_type: str):
    """
    根據 model_type 回傳對應模型。
    """
    if model_type == 'custom':
        return CNN5d()
    elif model_type == 'resnet18':
        return get_resnet18(num_classes=2, pretrained=True)
    elif model_type == 'resnet50':
        return get_resnet50(num_classes=2, pretrained=True)

    #########################################
    # 新增 Vision Transformer branch
    #########################################
    elif model_type == 'vit_b16':
        return get_vit_b16(num_classes=2, pretrained=True)

    else:
        raise ValueError(f"不支援的 model_type: {model_type}")
