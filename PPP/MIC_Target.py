import pandas as pd
import numpy as np
from minepy import MINE


def calculate_mic(df, target_column):
    # 创建一个 MINE 对象
    mine = MINE()

    # 存储每对变量的 MIC 值
    mic_values = {}

    # 计算每个变量和目标变量之间的 MIC 值
    for column in df.columns:
        if column != target_column:
            mine.compute_score(df[column].values, df[target_column].values)
            mic_values[column] = mine.mic()

    return mic_values


def calculate_pairwise_mic(df):
    # 创建一个 MINE 对象
    mine = MINE()

    # 存储每对变量的 MIC 值
    pairwise_mic_values = {}

    # 计算所有变量对之间的 MIC 值
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            mine.compute_score(df[columns[i]].values, df[columns[j]].values)
            pairwise_mic_values[(columns[i], columns[j])] = mine.mic()

    return pairwise_mic_values


def select_top_variables(mic_values, num_variables):
    # 根据MIC值排序并选择前num_variables个变量
    sorted_variables = sorted(mic_values.items(), key=lambda x: x[1], reverse=True)
    top_variables = [var for var, mic in sorted_variables[:num_variables]]

    return top_variables


# 示例：读取数据集
df = pd.read_csv('/home/home_new/chensf/WorkSpace/TII/EP#1234_Get/EP#1.csv')

# 选择目标变量列（比如最后一列）
target_column = df.columns[-1]

# 计算与目标变量的MIC
mic_values = calculate_mic(df, target_column)

# 打印与目标变量的MIC值
print("与目标变量的MIC值：")
for var, mic in mic_values.items():
    print(f"{var}: {mic}")


# 计算所有变量对之间的MIC值
#pairwise_mic_values = calculate_pairwise_mic(df)

# 打印所有变量对的MIC值
#print("\n所有变量对之间的MIC值：")
#for (var1, var2), mic in pairwise_mic_values.items():
#    print(f"{var1} - {var2}: {mic}")

# 选择MIC值排前的变量（比如选择前5个）
#num_variables = 5
#top_variables = select_top_variables(mic_values, num_variables)

#print(f"\n选择的前{num_variables}个变量是：{top_variables}")
