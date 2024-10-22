import torch
from torch import nn


class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)



    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = self.sigmoid(out).squeeze()

        return out


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Conv_Block(in_ch=3, out_ch=16, stride=2)
        self.layer2 = Conv_Block(in_ch=16, out_ch=32, stride=2)
        self.layer3 = Conv_Block(in_ch=32, out_ch=64, stride=2)
        self.layer4 = Conv_Block(in_ch=64, out_ch=128, stride=2)
        self.layer5 = Conv_Block(in_ch=128, out_ch=256, stride=2)
        self.layer6 = Conv_Block(in_ch=256, out_ch=512, stride=2)
        self.classifier = Classifier(input_dim=512)

    def forward(self, x):
        out = self.layer1(x)  # [128]
        out = self.layer2(out)  # [64]
        out = self.layer3(out)  # [32]
        out = self.layer4(out)  # [16]
        out = self.layer5(out)  # [8]
        out = self.layer6(out)  # [4]
        out = self.classifier(out)

        return out