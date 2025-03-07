import matplotlib.pyplot as plt
import torch
from lstmmodel import model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_loader import train_labels
# 加载测试集的输入CSV文件
df_test = pd.read_csv('test_input_sample.csv', header=None)
df_label=pd.read_csv('test_output_sample.csv',header=None)
# 提取日期和特征列
dates = df_test.iloc[:, 0]
features = df_test.iloc[:, 1:].values
labels=df_label.iloc[:,1:].values
# 对特征进行归一化处理
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_labels=scaler.fit_transform(labels)
# 定义综合指标的权重
r2_weight = -0.200
mae_weight = 0.200
rmse_weight = 0.600
def sliding_windows(data1, data2, seq_length):
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
length=5
scaled_features,scaled_labels=sliding_windows(scaled_features,scaled_labels,length)
# 转换为PyTorch张量
scaled_features = torch.tensor(scaled_features, dtype=torch.float32)#([989, 10, 12])
scaled_features = scaled_features.view(1000-length-1, length, 12)
scaled_labels = torch.tensor(scaled_labels, dtype=torch.float32)#([989, 10, 12])
scaled_labels = scaled_labels.view(-1, 1)
# 加载保存的模型参数
model.load_state_dict(torch.load('best_model.pth'))

# 设置模型为评估模式
model.eval()
with torch.no_grad():
   # 预测测试集的结果
   predictions = model(scaled_features)
print(predictions)
plt.plot( predictions, label='predictions')
plt.plot(scaled_labels, label='true')
plt.xlabel('time')
plt.ylabel('y')
plt.legend()
plt.show()
predictions1 = predictions.reshape(-1, 1)
df1=scaler.inverse_transform(predictions1) # 逆标准化
#df1=predictions1.detach().numpy()
#df1_dataframe=pd.DataFrame(df1)
# 将预测结果转换为DataFrame
df1_dataframe = pd.DataFrame(df1, columns=['预测结果'])
df1_dataframe['日期'] = dates
pred_dataframe = df1_dataframe[['预测结果','日期']]
df1_dataframe.to_csv('预测结果.csv')

plt.plot( df1, label='predictions_inverse')
plt.plot(labels, label='true_inverse')
plt.xlabel('time')
plt.ylabel('y')
plt.legend()
plt.show()
# 计算MAE指标
r2 = r2_score(scaled_labels.cpu().detach().numpy(), predictions1.cpu().detach().numpy())
mae = mean_absolute_error(scaled_labels.cpu().detach().numpy(), predictions1.cpu().detach().numpy())
rmse = np.sqrt(mean_squared_error(scaled_labels.cpu().detach().numpy(), predictions1.cpu().detach().numpy()))

# 计算综合指标
metric_test= r2_weight * r2 + mae_weight * mae + rmse_weight * rmse
print(r2,mae,rmse,metric_test)
