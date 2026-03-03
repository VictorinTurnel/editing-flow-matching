import torch
import torch.nn as nn
from torchvision import models

class TimeCondResNet18(nn.Module):
    
    def __init__(self, num_classes = 10):
        super(TimeCondResNet18, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        temp_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        with torch.no_grad():
            self.resnet.conv1.weight[:, :3, :, :] = temp_conv.weight
            self.resnet.conv1.weight[:, 3, :, :] = 0.

    def forward(self, x, t):

        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        input_res = torch.cat([x, t_map], dim=1)
        return self.resnet(input_res)
