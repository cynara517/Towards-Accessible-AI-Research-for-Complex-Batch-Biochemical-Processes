import pandas as pd
import numpy as np
from fcmeans import FCM 
# 读取数据集
df = pd.read_csv('/Users/shifanchen/Documents/WorkSpace/TII/EP#1234_Get/EP_Batch_270.csv')

data = df[['X15']].values

# 初始化 FCM 聚类模型，设定簇的个数 (m为批次数)
m = 270  # 例如，设定为EP的批次数
fcm = FCM(n_clusters=m)

# 拟合数据
fcm.fit(data)

# 获取每个数据点的隶属度
membership = fcm.u

# 对每个数据点，选择隶属度最高的簇
labels = np.argmax(membership, axis=1)

# 将结果添加到原始DataFrame中
df['Cluster_Label'] = labels

# 将聚类结果保存到新的CSV文件
df.to_csv('clustered_data.csv', index=False)

# 输出每个数据点所属的簇
print(df[['te', 'Cluster_Label']])
