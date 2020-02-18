# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 21:22
# @Author  : XiaoMa（小马）
# @qq      : 1530253396（任何问题欢迎联系）
# @File    : model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN1(nn.Module):
    def __init__(self,n_feature = 5, n_hidden = 10//1, n_output = 2):
        super(NN1,self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        x = torch.sigmoid(x)
        return x
class NN2(nn.Module):
    def __init__(self,n_feature = 5, n_hidden = 10, n_output = 2):
        super(NN2,self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
class NN3(nn.Module):
    def __init__(self,n_feature = 5, n_hidden = 10 // 2, n_output = 2):
        super(NN3,self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x
# net = MLP(n_feature=6, n_hidden=10, n_output=2)
class LR(torch.nn.Module):
    def __init__(self,n_feature = 5, n_output = 2):
        super(LR,self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


