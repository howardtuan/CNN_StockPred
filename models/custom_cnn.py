# my_project/my_project/models/custom_cnn.py

import torch
import torch.nn as nn

IMG_HEIGHT_SELECTED = 48
IMG_WIDTH_SELECTED = 29

class CNN5d(nn.Module):
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN5d, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), padding=(2, 1), stride=(1, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv1.apply(self.init_weights)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1), stride=(1, 1)),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2.apply(self.init_weights)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv3.apply(self.init_weights)

        # 第四層卷積 (conv4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4.apply(self.init_weights)

        # 重新計算 flatten 大小
        dummy_input = torch.zeros(1, 1, IMG_HEIGHT_SELECTED, IMG_WIDTH_SELECTED)
        with torch.no_grad():
            out = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
            flattened_size = out.view(1, -1).shape[1]

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(flattened_size, 2)
        self.fc.apply(self.init_weights)

        # 若用 CrossEntropyLoss，可考慮拿掉 softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)   # (N, 48, 29) → (N, 1, 48, 29)
        elif x.ndim == 5:
            x = x.squeeze(2)     # (N, 1, 1, 48, 29) → (N, 1, 48, 29)

        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)              # 新增這行：通過第四層
        x = x.view(x.size(0), -1)      # flatten
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
