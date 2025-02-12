# my_project/my_project/models/pretrained.py

import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=2, pretrained=True, freeze_backbone=True):
    net = models.resnet18(pretrained=pretrained)
    in_features = net.fc.in_features
    
    if freeze_backbone:
        # 凍結 backbone (conv, bn, layer1~4等)
        for param in net.parameters():
            param.requires_grad = False
    
    # 只將最後一層 (fc) 改成想要的輸出數量
    net.fc = nn.Linear(in_features, num_classes)
    
    # 若 freeze_backbone=True，只有 net.fc.* 參數會參與訓練
    return net

def get_resnet50(num_classes=2, pretrained=True):
    net = models.resnet50(pretrained=pretrained)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net

##############################################
# 新增 Vision Transformer: ViT-B/16
##############################################
def get_vit_b16(num_classes=2, pretrained=True, freeze_backbone=True):
    """
    回傳 torchvision 實作的 ViT-B/16 模型，最後一層改成 num_classes。
    需要 torchvision >= 0.14。
    """
    net = models.vit_b_16(pretrained=pretrained)
    in_features = net.heads.head.in_features

    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False

    net.heads.head = nn.Linear(in_features, num_classes)
    return net