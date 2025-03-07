import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import random

# 加载数据
data = pd.read_csv('traindata.csv')

# 提取特征和标签
#iloc函数用法：注意列不包含最后一列的输入，1:13，即为第2列到第12列；time：第0列；特征从第1列输入；结束为标号为12的列，+1为13
features = data.iloc[:, 1:13].values  # 提取前0-12列作为特征,shape(9000,12),float64
labels = data.iloc[:, 13].values  # 提取第13列作为标签,shape(9000,),float64

#testing_set1 = np.array(testing_set.iloc[:, 1:12])
#print(labels.shape)
# 归一化处理
scaler = StandardScaler()
features = scaler.fit_transform(features)#shape(9000,12)
labels = scaler.fit_transform(labels.reshape(-1,1))#这里不reshape会报错，按照提示信息reshape,shape(9000,1)
#print(labels.shape)

# 划分训练集和测试集,不用进行样本集的打乱
train_inputs = features[:7000]#(7000,12)
train_labels = labels[:7000]#(7000,1)
val_inputs = features[7000:]#(2000,12)
val_labels = labels[7000:]#(2000,1)

#数据滑窗

def sliding_windows(data1,data2, seq_length):
    x = []
    y = []

    # 这里默认前12个为特征，预测的目标在数据集第13个（最后）位置
    for i in range(len(data1) - seq_length - 1):
        _x = data1[i:(i + seq_length), :]
        _y = data2[i + seq_length, :]  # ‘-1’表示最后一个
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

#调用滑动窗口函数
length=7
train_inputs,train_labels=sliding_windows(train_inputs,train_labels,length)
val_inputs,val_labels=sliding_windows(val_inputs,val_labels,length)
# 转换为PyTorch张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)#([6989, 10, 12])
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)#([1989, 10, 12])
val_labels = torch.tensor(val_labels, dtype=torch.float32)#([1989, 1])

# 调整输入张量的形状以适应LSTM网络shape==(seq_length,batch_size,input_size)
train_inputs = train_inputs.view(7000-length-1, length, 12)
val_inputs = val_inputs.view(2000-length-1, length, 12)
train_labels = train_labels.view(-1,1)
val_labels = val_labels.view(-1,1)#[1989, 1]
print(val_labels.detach().numpy())


# 创建数据加载器
#先把训练、验证的分别搞成一个数据集
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)
#dataset, batch_size=16, shuffle=False,  num_workers=3
train_loader = DataLoader(train_dataset,batch_size=16, shuffle=True,  num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True,  num_workers=0)