import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def scores(y_true, y_predict):
    '''
    计算时间序列生成任务的评价指标
    :param y_true: 真实标签（目标时间序列）
    :param y_predict: 预测标签（生成的时间序列）
    :return: 评价指标字典，包括均方误差（MSE）、平均绝对误差（MAE）和R²系数
    '''
    # 计算均方误差（MSE）
    mse = mean_squared_error(y_true, y_predict)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(y_true, y_predict)

    # 计算R²系数（决定系数）
    r2 = r2_score(y_true, y_predict)

    return {
        'mse': mse,  # 均方误差
        'mae': mae,  # 平均绝对误差
        'r2': r2  # R²系数
    }


# 如果 y_true 和 y_predict 是 PyTorch 张量，可以直接使用下面的函数
def tensor_scores(y_true, y_predict):
    '''
    计算PyTorch张量形式的时间序列生成任务评价指标
    :param y_true: 真实标签（PyTorch张量）
    :param y_predict: 预测标签（PyTorch张量）
    :return: 评价指标字典，包括均方误差（MSE）、平均绝对误差（MAE）和R²系数
    '''
    y_true = y_true.cpu().numpy()  # 将PyTorch张量转为NumPy数组
    y_predict = y_predict.cpu().numpy()

    return scores(y_true, y_predict)
