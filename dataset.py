# my_project/my_project/dataset.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from .config import Config
from .transforms import get_transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file):
        """
        根據 Config.MODEL_TYPE 選擇 transforms 及讀取模式 (RGB or Gray)
        """
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file, sep='\t')

        self.model_type = Config.MODEL_TYPE
        self.transform = get_transforms(self.model_type)

        # 如果是自訂模型 => 灰階
        # 如果是 torchvision 預訓練 => RGB
        if self.model_type == 'custom':
            self.to_color = False
        else:
            self.to_color = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self._get_valid_image_path(idx)
        image = self._load_image(img_path)
        label = int(self.labels.iloc[idx, 1])
        return image, label

    def _get_valid_image_path(self, idx):
        base_name = self.labels.iloc[idx, 0]
        base_path = os.path.join(self.img_dir, base_name)

        if os.path.exists(base_path):
            return base_path

        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            test_path = base_path + ext
            if os.path.exists(test_path):
                return test_path
        return base_path  # 若仍找不到，就回傳原始路徑

    def _load_image(self, path):
        try:
            if self.to_color:
                # RGB
                img = Image.open(path).convert('RGB')
            else:
                # 灰階 (1-bit)
                img = Image.open(path).convert('1')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 建立空白圖
            if self.to_color:
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                img = Image.new('1', (Config.IMG_WIDTH[5], Config.IMG_HEIGHT[5]), 0)

        # 套用 transform
        try:
            return self.transform(img)
        except Exception as e:
            print(f"Error transforming image {path}: {e}")
            if self.to_color:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                return self.transform(Image.new('1', (Config.IMG_WIDTH[5], Config.IMG_HEIGHT[5]), 0))
