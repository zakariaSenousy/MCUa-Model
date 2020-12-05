import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BaseNetwork1(nn.Module):
    def __init__(self, name, channels=1):
        super(BaseNetwork1, self).__init__()
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



class PatchWiseNetwork1(BaseNetwork1):
    def __init__(self, channels=1):
        super(PatchWiseNetwork1, self).__init__('pw' + str(channels), channels)
        
        print('DenseNet Network Scale III')
        densenet = torchvision.models.densenet161(pretrained=True)
        densenet.classifier = nn.Sequential(nn.Linear(2208, 4))
        #self.features = nn.Sequential(*list(densenet.children())[:-1])        
        
        self.features = densenet.features
        self.classifier = densenet.classifier
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        
        ct = 0
        for child in self.features.children():
            ct += 1
            if ct < 11:
                for param in child.parameters():
                    param.requires_grad = False
        
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x



class ImageWiseNetwork1(BaseNetwork1):
    def __init__(self, channels=1):
        super(ImageWiseNetwork1, self).__init__('iw' + str(channels), channels)

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
        self.fc1 = nn.Linear(1 * 336 * 80, 128)
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
