# my_project/my_project/main.py

import os
import torch
from torch.utils.data import DataLoader, random_split

from .config import Config, setup_gpu
from .dataset import CustomImageDataset
from .models.factory import get_model
from .train import train_model
from .evaluate import evaluate_model, load_model

def main():
    # 從 Config 讀取
    img_dir = Config.IMAGE_DIR
    labels_file = Config.LABEL_FILE
    pth_name = Config.MODEL_PATH

    print(f"圖片路徑: {img_dir}")
    print(f"標籤路徑: {labels_file}")
    print(f"模型檔名: {pth_name}")
    print(f"模型類型: {Config.MODEL_TYPE}")

    device = setup_gpu()
    print(f"使用裝置: {device}")
    torch.cuda.empty_cache()

    # 1) 讀取資料
    dataset = CustomImageDataset(img_dir=img_dir, labels_file=labels_file)

    # 2) 資料切分
    if Config.USE_RANDOM_SPLIT:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        train_dataset, test_dataset = dataset, dataset

    # 3) 建立 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # 4) 取得模型 (factory)
    model = get_model(Config.MODEL_TYPE)

    # 5) 若檔案存在，就嘗試載入
    if os.path.exists(pth_name):
        try:
            loaded_model, opt_state, loaded_epoch, loaded_loss = load_model(model, pth_name)
            model = loaded_model
            print(f"已載入模型：Epoch={loaded_epoch}, Loss={loaded_loss}")
        except Exception as e:
            print(f"載入模型失敗：{e}，將重新訓練。")
    else:
        print("找不到已儲存的模型檔，將重新訓練。")

    # 多GPU
    if Config.USE_DATAPARALLEL and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 6) 訓練
    print("開始訓練模型...")
    train_model(model, train_loader, test_loader, device, pth_name=pth_name)

    # 7) 評估
    print("評估模型...")
    evaluate_model(model, test_loader, device)
