# my_project/my_project/evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from . import line_bot_helper
from .config import Config

def print_training_log(epoch, train_loss, test_loss, device, early_stopping_counter):
    used_mem = 0
    if device.type == 'cuda':
        used_mem = torch.cuda.memory_allocated(device) / 1024**2
    print(f"GPU mem used: {used_mem:.1f}MB")
    print(f"Epoch [{epoch+1}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Early Stopping Counter: {early_stopping_counter}\n")


def plot_confusion_matrix(all_labels, all_preds):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_probability_distribution(probs, labels):
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [probs[labels == 0][:, 1], probs[labels == 1][:, 1]],
        labels=['Class 0', 'Class 1']
    )
    plt.title('Distribution of Probability for Each Class')
    plt.ylabel('Probability of Predicting "1"')
    plt.show()


def evaluate_model(model, test_loader, device, plot_roc=True):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # [batch, 2]
            probs = outputs
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 混淆矩陣
    plot_confusion_matrix(all_labels, all_preds)

    # 分類報告
    print("\n分類結果 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

    # 機率分布統計
    print("\n機率分布統計:")
    for cls in [0, 1]:
        cls_probs = all_probs[all_labels == cls]
        mean_prob = cls_probs.mean(axis=0)
        std_prob = cls_probs.std(axis=0)
        print(f"  - Class {cls}: 平均={mean_prob}, 標準差={std_prob}")

    # 機率分布圖
    plot_probability_distribution(all_probs, all_labels)

    # ROC
    if plot_roc:
        plot_roc_curve(all_labels, all_probs[:, 1])

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n模型 Accuracy: {accuracy:.4f}")

    # line bot 訊息
    lineSTR = f"{Config.MODEL_TYPE} 模型訓練完成！記得檢查結果~"
    line_bot_helper.send_message_to_group(lineSTR, accuracy)

    return accuracy

# 這裡也放 save_model / load_model，或你可放 utils.py
def save_model(model, optimizer, epoch, test_loss, filename=None):
    if filename is None:
        filename = Config.MODEL_PATH
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")


def load_model(model, filename=None):
    import os
    if filename is None:
        filename = Config.MODEL_PATH

    if not os.path.exists(filename):
        print(f"檔案 {filename} 不存在，無法載入。")
        return model, None, None, None

    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model

    checkpoint = torch.load(filename)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    return (
        base_model,
        checkpoint.get('optimizer_state_dict'),
        checkpoint.get('epoch'),
        checkpoint.get('loss')
    )