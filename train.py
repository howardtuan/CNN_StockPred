# my_project/my_project/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from .config import Config
from .evaluate import print_training_log, save_model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}], "
                  f"Batch [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")
            
    return running_loss / len(train_loader)

def validate_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    return test_loss / len(test_loader)

def early_stopping_check(current_loss, best_loss, counter):
    return (counter >= Config.EARLY_STOPPING_PATIENCE)

def train_model(model, train_loader, test_loader, device, pth_name=None):
    """
    主訓練流程。
    pth_name: 要儲存的檔名（若不指定則可從 Config.MDOEL_PATH 讀）
    """
    if pth_name is None:
        pth_name = Config.MODEL_PATH

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss = validate_epoch(model, test_loader, criterion, device)

        scheduler.step(test_loss)
        print_training_log(epoch, train_loss, test_loss, device, early_stopping_counter)

        if early_stopping_check(test_loss, best_loss, early_stopping_counter):
            print(f"早停觸發：連續 {early_stopping_counter} 次沒有改善，停止訓練。")
            break

        if test_loss < best_loss - Config.MIN_DELTA:
            best_loss = test_loss
            early_stopping_counter = 0
            # 儲存最佳模型
            save_model(model, optimizer, epoch, test_loss, filename=pth_name)
        else:
            early_stopping_counter += 1
