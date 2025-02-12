# my_project/my_project/transforms.py

from torchvision import transforms as T
from .config import Config

def get_transforms_for_custom_cnn():
    # 自訂 CNN (灰階, 48x29)
    return T.Compose([
        T.Resize((Config.IMG_HEIGHT[5], Config.IMG_WIDTH[5])),
        T.ToTensor(),
    ])

def get_transforms_for_pretrained():
    # 一般預訓練 (ResNet, ViT...) => RGB 224x224 + Normalize
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])  # ImageNet 常見
    ])

def get_transforms(model_type):
    if model_type == 'custom':
        return get_transforms_for_custom_cnn()
    else:
        return get_transforms_for_pretrained()
