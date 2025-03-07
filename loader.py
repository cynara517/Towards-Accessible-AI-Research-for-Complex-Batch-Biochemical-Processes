# -*- coding: utf-8 -*-
import os
import torch.utils.data as Data
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import opt
from torch.utils.data.sampler import WeightedRandomSampler
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition

# 注意：这里不再在 loader 文件中直接转移到 GPU
device = opt.device  # 可保留，但不在数据加载时使用

def transform_dataset(x_data, y_data, n_input, n_output):
    """
    将数据转换为时序预测的格式：
      - 输入为连续 n_input 步，每步包含所有特征；
      - 输出为接下来的 n_output 步，每步包含所有特征。
    """
    # 如果 y_data 是一维数组，将其转换为二维数组（列向量）
    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)

    data_size = x_data.shape[0]
    num_samples = data_size - n_input - n_output + 1
    X = np.empty((num_samples, n_input, x_data.shape[1]))
    Y = np.empty((num_samples, n_output, y_data.shape[1]))

    for i in range(num_samples):
        X[i] = x_data[i:i + n_input, :]
        Y[i] = y_data[i + n_input:i + n_input + n_output, :]

    return X, Y

def get_data(path, step=512):
    # 读取数据
    train = pd.read_csv(path)
    train.columns = ["Batch", "Time(h)", "fi", "wi", "nd", "ht", "zt", "cer", "hx"]

    # 选择所有特征
    train_data = train.iloc[:, 2:].to_numpy().astype(float)  # shape: (样本数, 特征数)

    # 归一化
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)

    # 注意：目标也使用所有特征
    train_target = train_data.copy()

    # 划分训练、测试数据
    sss = ShuffleSplit(n_splits=1, test_size=opt.testRatio, random_state=0)
    for train_index, test_index in sss.split(train_data):
        X_train_full, X_test_full = train_data[train_index], train_data[test_index]
        y_train_full, y_test_full = train_target[train_index], train_target[test_index]

    # 划分训练集和验证集
    sss = ShuffleSplit(n_splits=1, test_size=opt.valRatio, random_state=0)
    for train_index, valid_index in sss.split(X_train_full):
        X_train, X_valid = X_train_full[train_index], X_train_full[valid_index]
        y_train_full, y_valid_full = y_train_full[train_index], y_train_full[valid_index]

    # 将数据转换为时序格式，输入和输出都为窗口长度 step
    X_train, y_train = transform_dataset(X_train, y_train_full, step, step)
    X_valid, y_valid = transform_dataset(X_valid, y_valid_full, step, step)
    X_test, y_test = transform_dataset(X_test_full, y_test_full, step, step)

    # 转换为 tensor（不调用 .to(device)）
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_loader(path, step=512, batch_size=128, num_workers=4):
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(path, step)
    return __dataLoader(X_train, y_train, batch_size, num_workers), \
           __dataLoader(X_valid, y_valid, batch_size, num_workers), \
           __dataLoader(X_test, y_test, batch_size, num_workers)

def __dataLoader(X, Y, batch_size, num_workers=4):
    # 保持数据在 CPU 上，不调用 .to(device)
    train_dataset = Data.TensorDataset(X, Y)
    train_loader = Data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

def get_predict_loader(path, step=512, batch_size=128, num_workers=4):
    train = pd.read_csv(path)
    train.columns = ["Batch", "Time(h)", "fi", "wi", "nd", "ht", "zt", "cer", "hx"]

    train_data = train.iloc[:, 2:].to_numpy().astype(float)
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    # 目标也使用所有特征，并且步长设置为与输入相同
    X_all, y_all = transform_dataset(train_data, train_data, step, step)

    X_all = torch.tensor(X_all, dtype=torch.float32)
    y_all = torch.tensor(y_all, dtype=torch.float32)

    dataset = Data.TensorDataset(X_all, y_all)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader

if __name__ == "__main__":
    # Test the data loading
    get_data(r"data\EP#1.csv")
