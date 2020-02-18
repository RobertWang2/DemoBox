# -*- coding: utf-8 -*-
# @Time    : 2020/2/12 21:38
# @Author  : XiaoMa（小马）
# @qq      : 1530253396（任何问题欢迎联系）
# @File    : baseline.py
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.model import NN1, NN2, NN3, LR
class ExpertSystem():
    '''
    专家系统
    '''
    def __init__(self, modelNameList):
        self.epoch = 1000
        self.data = pd.read_csv('../data/记录数据2.csv',engine='python')
        self.y = self.data['regular'].values
        self.x = self.data.drop(['regular'],axis=1).values
        del self.data
        self._preprocess()
        self.modelNameList = modelNameList
        self.getmodels()
    def _preprocess(self):
        sc = StandardScaler()
        self.x = sc.fit_transform(self.x)
        self.x1, self.y1 = self.x[0:1000], self.y[0:1000]
        self.x2, self.y2 = self.x[1000:2000], self.y[1000:2000]
        self.x3, self.y3 = self.x[2000:3000], self.y[2000:3000]
        self.x4, self.y4 = self.x[3000:4000], self.y[3000:4000]
        self.x, self.y = Variable(torch.Tensor(self.x).type(torch.FloatTensor)), Variable(torch.Tensor(self.y).type(torch.LongTensor))
        self.x1, self.y1 = Variable(torch.Tensor(self.x1).type(torch.FloatTensor)), Variable(torch.Tensor(self.y1).type(torch.LongTensor))
        self.x2, self.y2 = Variable(torch.Tensor(self.x2).type(torch.FloatTensor)), Variable(torch.Tensor(self.y2).type(torch.LongTensor))
        self.x3, self.y3 = Variable(torch.Tensor(self.x3).type(torch.FloatTensor)), Variable(torch.Tensor(self.y3).type(torch.LongTensor))
        self.x4, self.y4 = Variable(torch.Tensor(self.x4).type(torch.FloatTensor)), Variable(torch.Tensor(self.y4).type(torch.LongTensor))

    def getmodels(self):
        self.models = []
        self.model_dict =  {'NN1':NN1(), 'NN2':NN2(), 'NN3':NN3(), 'LR':LR()}
        for idx,name in enumerate(self.modelNameList):
            model = self.model_dict.get(name)
            print(idx, model)
            self.models.append(model)

    def train(self, lr = 0.012):
        self.pred = []
        self.out = []
        for idx, model in enumerate(self.models):
            print(f'train {idx + 1} model: ')
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.012)
            # model.train()
            for i in range(self.epoch):
                if idx == 0:
                    out = model(self.x1)
                    loss = loss_func(out, self.y1)
                elif idx == 1:
                    out = model(self.x2)
                    loss = loss_func(out, self.y2)
                elif idx == 2:
                    out = model(self.x3)
                    loss = loss_func(out, self.y3)
                elif idx == 3:
                    out = model(self.x4)
                    loss = loss_func(out, self.y4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    self.eval(idx, i, out, loss)
            # model.eval()
            if idx == 0:
                out = model(self.x1)
            elif idx == 1:
                out = model(self.x2)
            elif idx == 2:
                out = model(self.x3)
            elif idx == 3:
                out = model(self.x4)
            self.out.extend(out.max(-1)[1].detach().numpy())
            self.pred.extend(out.max(-1)[1].numpy())
        print(sum(self.pred))
        print(f'专家系统的acc为：{torch.mean((torch.Tensor(self.pred) == self.y[:4000]).float())}')

    def eval(self, idx, i, out, loss):
        t = None
        if idx == 0:
            t = out.max(-1)[1] == self.y1
        elif idx == 1:
            t = out.max(-1)[1] == self.y2
        elif idx == 2:
            t = out.max(-1)[1] == self.y3
        elif idx == 3:
            t = out.max(-1)[1] == self.y3
        pred = out.max(-1)[1]
        acc = torch.mean(t.float())
        print(f'Epoch: {i}, loss: {loss}, acc: {acc}')
from torch.utils.data import Dataset, DataLoader
class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        x = self.data.iloc[item,:self.data.shape[1] - 1].values.astype('float32')
        y = self.data.iloc[item,-1]
        return x, y
class TransferLearning():

    def __init__(self):
        self.epoch = 1000
        self.data = pd.read_csv('../data/记录数据2.csv', engine='python')
        self.newdata = pd.read_csv('../data/sleepdata2.csv', engine='python')
        self.newdata = pd.concat([self.data, self.newdata])
        self.y = self.data['regular'].values
        self.x = self.data.drop(['regular'], axis=1).values
        self.newy = self.newdata['regular'].values
        self.newx = self.newdata.drop(['regular'], axis=1).values
        self._preprocess()
        self.model = LR()

    def _preprocess(self):
        sc = StandardScaler()
        self.x = sc.fit_transform(self.x)
        self.x, self.y = Variable(torch.Tensor(self.x).type(torch.FloatTensor)), Variable(torch.Tensor(self.y).type(torch.LongTensor))
        self.newx = sc.fit_transform(self.newx)
        self.newx, self.newy = Variable(torch.Tensor(self.newx).type(torch.FloatTensor)), Variable(torch.Tensor(self.newy).type(torch.LongTensor))

    def train(self, init = True):

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.012)
        self.pred = []
        for i in range(self.epoch):
            out = self.model(self.newx)
            loss = loss_func(out, self.newy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                self.eval(i,out,loss)
        self.out = out
        self.pred.extend(out.max(-1)[1].numpy())
        torch.save(self.model.state_dict(), '../models/TransferLearning.pth')
        print(f'不加迁移学习的acc为：{torch.mean((torch.Tensor(self.pred) == self.newy).float())}')
        ckp = torch.load('../models/TransferLearning.pth')
        self.model.load_state_dict(ckp)
        self.pred = []
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.012)
        # self.model.train()
        # self.newx = self.newdata
        for i in range(self.epoch):
            out = self.model(self.x)
            loss = loss_func(out, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                # self.model.eval()
                # out = self.model(self.newx)
                t = out.max(-1)[1] == self.y
                acc = torch.mean(t.float())
                print(f'Epoch: {i}, loss: {loss}, acc: {acc}')
                # self.eval(i, out, loss)
        self.out = out.detach().numpy()
        self.out = self.model(self.x).detach().numpy()
        self.pred.extend(out.max(-1)[1])
        print(f'加上迁移学习的acc为：{torch.mean((torch.Tensor(self.pred[100:]) == self.y[100:]).float())}')
        self.pred = self.out.max(-1)[1]

    def eval(self, i, out, loss):
        t = out.max(-1)[1] == self.newy
        pred = out.max(-1)[1]
        acc = torch.mean(t.float())
        print(f'Epoch: {i}, loss: {loss}, acc: {acc}')

expertSystem = ExpertSystem(['NN1','NN2','NN3','LR'])
expertSystem.train()
transferLearning = TransferLearning()
transferLearning.train(init = False)
out = np.array(expertSystem.out).reshape(-1,1) * transferLearning.out[:4000]
ensumble = out.argmax(axis = 1)
res = ensumble == transferLearning.y.detach().numpy()[:4000]
print('专家系统+门控，迁移学习的结果为：', res.mean())
