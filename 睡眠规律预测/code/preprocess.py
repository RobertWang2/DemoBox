# -*- coding: utf-8 -*-
# @Time    : 2020/2/12 21:38
# @Author  : XiaoMa（小马）
# @qq      : 1530253396（任何问题欢迎联系）
# @File    : baseline.py
# os.1 数据处理
import pandas as pd
class FirstDataPreprocess:
    '''
    数据处理思路：
    1.start和end，睡眠质量与日期无关，所以去掉这部分，分钟转小时制，如8：30 -> 8.5
    2.Time in bed 同1
    3.根据Sleep score确定睡眠质量（80~100是好，60~79是一般，0~59）
    4.数据归一化（需要后序处理）
    5.删除无用字段
    '''
    def __init__(self,basepath):
        self.base_path = basepath
        self.data = pd.read_csv(base_path + '\sleepdata.csv',engine='python')
        self.preprocess()
    def _time_preprocess(self,row):
        time = row.split('--')[1].split(':')
        time = float(time[0]) + float(time[1]) / 60
        return time

    def _time_preprocess2(self, row):
        time = row.split(':')
        time = float(time[0]) + float(time[1]) / 60
        return time

    def _time_preprocess3(self, row):
        score = int(row[:-1])
        return score
    def _score_compute(self,row):
        '''
        0-59: 0
        60-79: 1
        80-100: 2
        注意：因为储存只需要很少的字节，如果数据过大，可以转更小的字节节省内存
        '''
        if row >= 75:
            return 1
        else:
            return 0

    def preprocess(self):
        self.data['Start'] = self.data['Start'].apply(self._time_preprocess)
        self.data['End'] = self.data['End'].apply(self._time_preprocess)
        self.data['Time in bed'] = self.data['Time in bed'].apply(self._time_preprocess2)
        self.data['Sleep score'] = self.data['Sleep score'].apply(self._time_preprocess3)
        self.data['sleep quality'] = self.data['Sleep score'].apply(self._score_compute)
        self.data = self.data.drop(['Sleep score','睡眠质量（好，一般，不好）'],axis = 1)
        print(self.data.head())
        self.data.to_csv('../data/sleepdata2.csv',index=False)

class SecondDataPreprocess:
    '''
    数据处理思路：
    1.Start 、Time in bed 和 End 处理
    2.求解睡眠质量

    '''
    def __init__(self,basepath):
        self.base_path = basepath
        self.data = pd.read_excel(base_path + '\记录数据.xlsx')
        self.preprocess()
    def _time_preprocess(self,row):
        # 1900-01-07 07:12:00
        row = str(row)
        if  '-' in row:
            time = row.split()[1].split(':')
        elif ':' in row:
            time = row.split(':')
        else:
            return float(row)
        # try:
        time = float(time[0]) + float(time[1]) / 60
        # except:
        #     print(time)
        return time

    def _score_compute(self,row):
        '''
        计算是否规律:
            看每一个特征的值是否在对应的区间之内[mean - 2*std, mean + 2*std]
        '''
        # res = 1
        if row['Start'] < self.Start_low or row['Start'] > self.Start_high or \
            row['End'] < self.End_low or row['End'] > self.End_high or \
            row['Time in bed'] < self.Time_in_bed_low or row['Time in bed'] > self.Time_in_bed_high or \
            row['Heart rate'] < self.Heart_rate_low or row['Heart rate'] > self.Heart_rate_high or \
            row['Activity (steps)'] < self.Activity_low or row['Activity (steps)'] > self.Activity_high or \
            row['Sleep score'] < self.Sleep_score_low or row['Sleep score'] > self.Sleep_score_high:
            return 0
        return 1
    def preprocess(self):
        self.data.columns = ['Start', 'End', 'Time in bed', 'Heart rate', 'Activity (steps)', 'Sleep score', 'regular']
        self.data['Start'] = self.data['Start'].apply(self._time_preprocess)
        self.data['End'] = self.data['End'].apply(self._time_preprocess)
        self.data['Time in bed'] = self.data['Time in bed'].apply(self._time_preprocess)
        # 计算特征的std 和 mean
        self.Start_std = self.data['Start'].std(axis=0)
        self.End_std = self.data['End'].std(axis=0)
        self.Time_in_bed_std = self.data['Time in bed'].std(axis=0)
        self.Heart_rate_std = self.data['Heart rate'].std(axis=0)
        self.Activity_std = self.data['Activity (steps)'].std(axis=0)
        self.Sleep_score_std = self.data['Sleep score'].std(axis=0)
        self.Start_mean = self.data['Start'].mean(axis=0)
        self.End_mean = self.data['End'].mean(axis=0)
        self.Time_in_bed_mean = self.data['Time in bed'].mean(axis=0)
        self.Heart_rate_mean = self.data['Heart rate'].mean(axis=0)
        self.Activity_mean = self.data['Activity (steps)'].mean(axis=0)
        self.Sleep_score_mean = self.data['Sleep score'].mean(axis=0)

        self.Start_low = self.Start_mean - 2 * self.Start_std
        self.Start_high = self.Start_mean + 2 * self.Start_std
        self.End_low = self.End_mean - 2 * self.End_std
        self.End_high = self.End_mean + 2 * self.End_std
        self.Time_in_bed_low = self.Time_in_bed_mean - 2 * self.Time_in_bed_std
        self.Time_in_bed_high = self.Time_in_bed_mean + 2 * self.Time_in_bed_std
        self.Heart_rate_low = self.Heart_rate_mean - 2 * self.Heart_rate_std
        self.Heart_rate_high = self.Heart_rate_mean + 2 * self.Heart_rate_std
        self.Activity_low = self.Activity_mean - 2 * self.Activity_std
        self.Activity_high = self.Activity_mean + 2 * self.Activity_std
        self.Sleep_score_low = self.Sleep_score_mean - 2 * self.Sleep_score_std
        self.Sleep_score_high = self.Sleep_score_mean + 2 * self.Sleep_score_std
        self.data['regular'] = self.data.apply(self._score_compute,axis=1)
        self.data = self.data.drop(['Sleep score'],axis=1)
        self.data.to_csv('../data/记录数据2.csv',index=False)

if __name__ == '__main__':
    base_path = r'E:\project\DemoBox\睡眠规律预测\data'
    # data = SecondDataPreprocess(base_path)
    data = FirstDataPreprocess(base_path)
