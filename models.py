import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import segmentation_models_pytorch as smp

def load_model(model_name, classes, dropout, pretrained):
    if model_name == 'resnet50':
        return Cloud_Resnet50(classes, dropout=dropout, pretrained=pretrained)
    elif model_name == 'resnet34':
        return Cloud_Resnet34(classes, dropout=dropout, pretrained=pretrained)
    elif model_name == 'resnet101':
        return Cloud_Resnet101(classes, dropout=dropout, pretrained=pretrained)
    elif model_name == 'densenet161':
        return Cloud_densenet161(classes, dropout=dropout, pretrained=pretrained)
    elif model_name == 'resnext50_32x4d':
        return Cloud_Resnext504d(classes, dropout=dropout, pretrained=pretrained)
    elif model_name == 'wide_resnet50_2':
        return Cloud_Wideresnet50(classes, dropout=dropout, pretrained=pretrained)
    elif model_name == 'mansnet':
        return Cloud_Mansnet(classes, dropout=dropout, pretrained=pretrained)
    else:
        raise Exception('Error Model Name!')

class Cloud_Resnet50(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_Resnet50,self).__init__()
        self.net = torch.nn.Sequential(*(list(models.resnet50(pretrained=pretrained).children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(4 * 2048, classes)
        if dropout is None:
            self.dropout = nn.Dropout(p=0)
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.net(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud_Resnet50'

class Cloud_Resnet34(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_Resnet34,self).__init__()
        self.net = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(4*512,classes)
        if dropout is None:
            self.dropout = nn.Dropout(p=0)
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.net(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud_Resnet34'

class Cloud_Resnet101(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_Resnet101,self).__init__()
        self.net = models.resnet101(pretrained=pretrained)
        self.net.avgpool = nn.AdaptiveAvgPool2d(2)
        self.net.fc = nn.Linear(4*2048,1000)
        self.fc = nn.Linear(1000,classes)
        if dropout is None:
            self.dropout = nn.Dropout(p=0)
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud_Resnet101'

class Cloud_Resnext504d(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_Resnext504d,self).__init__()
        self.net = torch.nn.Sequential(*(list(models.resnext50_32x4d(pretrained=pretrained).children())[:-2]))
        if dropout is None:
            self.dropout = nn.Dropout(0)
        else:
            self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(2048*4, classes)

    def forward(self,x):
        x = self.net(x)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud-Resnext101_32x8d'

class Cloud_densenet161(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_densenet161,self).__init__()
        self.net = torch.nn.Sequential(*(list(models.densenet161(pretrained=pretrained).children())[:-1]))
        if dropout is None:
            self.dropout = nn.Dropout(0)
        else:
            self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(2208*4, classes)

    def forward(self,x):
        x = self.net(x)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud-densenet161'

class Cloud_Wideresnet50(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_Wideresnet50,self).__init__()
        self.net = torch.nn.Sequential(*(list(models.wide_resnet50_2(pretrained=pretrained).children())[:-1]))
        if dropout is None:
            self.dropout = nn.Dropout(0)
        else:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, classes)

    def forward(self,x):
        x = self.net(x)
        x = x.squeeze()
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud-wide_resnet50'

class Cloud_Mansnet(nn.Module):
    def __init__(self,classes,dropout=None,pretrained=True):
        super(Cloud_Mansnet,self).__init__()
        self.net = torch.nn.Sequential(*(list(models.mnasnet1_0(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(4*1280,classes)
        if dropout is None:
            self.dropout = nn.Dropout(p=0)
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.net(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict(self,x):
        self.eval()
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x

    def getname(self):
        return 'Cloud_Mnasnet10'


if __name__ == '__main__':
    import os
    x = torch.randn((2,3,256,256))
    m = Cloud_Resnext504d(4,pretrained=False)
    print(m)
    y = m(x)
    # print(y.shape)

    # net = models.resnext101_32x8d(pretrained=True)
    # print(net)
    # net = models.wide_resnet101_2(pretrained=True)

    # model = load_model('resnext101',4,0.4,False)
    # model.load_state_dict(torch.load('model_test.pth'))
    # torch.save(model.state_dict(),'model_test.pth')

    # print(model)