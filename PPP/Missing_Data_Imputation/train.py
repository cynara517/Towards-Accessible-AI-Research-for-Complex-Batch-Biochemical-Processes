import torch
import torch.nn as nn
from lstmmodel import model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import pandas as pd
# 从 数据加载文件导入数据函数
from data_loader import train_loader, val_loader

for train_dataset in train_loader:
    train_inputs,train_labels=train_dataset

for val_dataset in val_loader:
    val_inputs,val_labels=val_dataset
# 定义综合指标的权重
r2_weight = -0.200
mae_weight = 0.200
rmse_weight = 0.600



criterion = nn.MSELoss()  # 使用MAE作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.002)

best_metric = float('inf')  # 初始化最佳综合指标为正无穷大
best_model = None  # 最佳模型参数的保存位置

num_epochs = 100
train_losses = []
val_losses = []
metrics = []

#模型训练
for epoch in range(num_epochs):
    # 训练模型
    model.train()
    train_loss = 0.0
    outputs = model(train_inputs)
    outputs = outputs.reshape(-1, 1)
    optimizer.zero_grad()
    train_loss = criterion(outputs, train_labels)
    train_loss.backward(retain_graph=True)
    optimizer.step()
    #记录损失函数值
    train_loss += train_loss.item()
    #计算平均损失
    train_loss /= len(train_inputs)
    train_losses.append(train_loss)
    loss_r2 = r2_score(train_labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1}/{num_epochs}, train Loss: {train_loss:.4f}, r2: {loss_r2:.4f}")
        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        metric = 0.0
        val_outputs = model(val_inputs)
        val_outputs = val_outputs.reshape(-1, 1)
        val_loss = criterion(val_outputs, val_labels)
        val_loss += val_loss.item()
        val_loss /= len(val_inputs)
        val_losses.append(val_loss)
        # 计算MAE指标
        r2 = r2_score(val_labels.cpu().detach().numpy(), val_outputs.cpu().detach().numpy())
        mae = mean_absolute_error(val_labels.cpu().detach().numpy(), val_outputs.cpu().detach().numpy())
        rmse = np.sqrt(mean_squared_error(val_labels.cpu().detach().numpy(), val_outputs.cpu().detach().numpy()))

        # 计算综合指标
        metric+= r2_weight * r2 + mae_weight * mae + rmse_weight * rmse
        metric /= len(val_inputs)
        metrics.append(metric)
        # 打印当前的损失、综合指标和轮次
        print(f"Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Metric: {metric:.4f}")

        # 检查综合指标是否小于最佳综合指标
        if metric < best_metric:
            best_metric = metric
            # 保存当前模型参数
            torch.save(model.state_dict(), 'best_model.pth')
            best_model = 'best_model.pth'
train_loss_all_np=[]
val_loss_all_np=[]
for i in range(num_epochs):
    train_loss_all_np.append(train_losses[i].detach().numpy())
    if num_epochs % 10 == 0:
     val_loss_all_np.append(val_losses[int(i/10)].detach().numpy())
plt.figure(figsize=(14, 5))
plt.subplot(1,2,1)
plt.plot(range(num_epochs), train_loss_all_np, label='Train Loss')
plt.plot(range(num_epochs), val_loss_all_np, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(int(num_epochs/10)), metrics, label='Metrics')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()