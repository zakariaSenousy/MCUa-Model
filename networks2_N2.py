import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BaseNetwork2(nn.Module):
    def __init__(self, name, channels=1):
        super(BaseNetwork2, self).__init__()
        self._name = name
        self._channels = channels

    def name(self):
        return self._name

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class PatchWiseNetwork2(BaseNetwork2):
    def __init__(self, channels=1):
        super(PatchWiseNetwork2, self).__init__('pw' + str(channels), channels)
        
        print('ResNet Network Scale II')
        resnet = torchvision.models.resnet152(pretrained=True)
        resnet.fc = nn.Sequential(nn.Linear(512*4, 4))
               
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = resnet.fc
        
    def forward(self, x):
        x = self.features(x)
        
        ct = 0
        for child in self.features.children():
            ct += 1
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x



class ImageWiseNetwork2(BaseNetwork2):
    def __init__(self, channels=1):
        super(ImageWiseNetwork2, self).__init__('iw' + str(channels), channels)

        self.features = nn.Sequential(
            # Block 1 #66 for old context aware
            nn.Conv2d(in_channels= 6 * channels , out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
                        

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
        )
        
        self.fc1 = nn.Linear(1 * 294 * 64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(p = 0.7)
        
       

        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        y = self.fc4(self.dropout(x))

        x = F.log_softmax((y), dim=1)
        return x
