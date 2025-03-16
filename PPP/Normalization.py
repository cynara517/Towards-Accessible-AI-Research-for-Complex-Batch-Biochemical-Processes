import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_statistics(df):
    # 计算每个Batch的统计信息
    stats = []

    # 获取所有Batch
    batch_list = df['Batch'].unique()

    for batch in batch_list:
        batch_data = df[df['Batch'] == batch]
        batch_stats = {'Batch': batch, 'Length': len(batch_data)}

        # 计算每列的最大值、最小值、均值、方差
        for column in batch_data.columns:
            if column != 'Batch' and column != 'Time(h)':  # 排除'Batch'和'Time(h)'
                batch_stats[f'{column}_max'] = batch_data[column].max()
                batch_stats[f'{column}_min'] = batch_data[column].min()
                batch_stats[f'{column}_mean'] = batch_data[column].mean()
                batch_stats[f'{column}_std'] = batch_data[column].std()

        stats.append(batch_stats)

    return pd.DataFrame(stats)


def z_normalize(df):
    # 对每个Batch进行Z归一化
    scaler = StandardScaler()
    columns_to_normalize = [col for col in df.columns if col != 'Batch' and col != 'Time(h)']  # 除了'Batch'和'Time(h)'列

    df_copy = df.copy()
    for batch in df['Batch'].unique():
        batch_data = df_copy[df_copy['Batch'] == batch]
        batch_data[columns_to_normalize] = scaler.fit_transform(batch_data[columns_to_normalize])
        df_copy.loc[df_copy['Batch'] == batch, columns_to_normalize] = batch_data[columns_to_normalize]

    return df_copy


def main():
    # 读取原始数据集
    df = pd.read_csv('data_after_MIC.csv')

    # 计算统计信息并保存到CSV文件
    stats_df = calculate_statistics(df)
    stats_df.to_csv('Batch_statistics.csv', index=False)

    # Z归一化并保存到新文件
    normalized_df = z_normalize(df)
    normalized_df.to_csv('processed.csv', index=False)

    print("统计信息和Z归一化已经成功处理并保存到CSV文件。")


if __name__ == "__main__":
    main()
