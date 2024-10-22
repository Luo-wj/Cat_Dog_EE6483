import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)



    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


class Res_Conv_Block_att(nn.Module):
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

        if self.stride == 2 or (self.in_ch != self.out_ch):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.cbam = CBAMBlock(channel=out_ch,reduction=4,kernel_size=3)


    def forward(self, x):
        if self.stride == 2 or (self.in_ch != self.out_ch):
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        
        out = self.cbam(out)

        return out



class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(input_dim, 10)

    def forward(self, x):
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out



class ResNet_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Res_Conv_Block_att(in_ch=3, out_ch=16, stride=1)
        self.layer2 = Res_Conv_Block_att(in_ch=16, out_ch=32, stride=2)
        self.layer3 = Res_Conv_Block_att(in_ch=32, out_ch=64, stride=1)
        self.layer4 = Res_Conv_Block_att(in_ch=64, out_ch=128, stride=2)
        self.layer5 = Res_Conv_Block_att(in_ch=128, out_ch=256, stride=1)
        self.layer6 = Res_Conv_Block_att(in_ch=256, out_ch=512, stride=2)
        
        self.classifier = Classifier(input_dim=512)

    def forward(self, x):
        out = self.layer1(x)  # [32]
        out = self.layer2(out)  # [16]
        out = self.layer3(out)  # [16]
        out = self.layer4(out)  # [8]
        out = self.layer5(out)  # [8]
        out = self.layer6(out)  # [4]

        out = self.classifier(out)

        return out