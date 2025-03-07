
import torch
from trainer_DP import Trainer
import argparse
import opt_DP as opt
from models.lstm_DP import BiLSTM
from utils.loader import get_predict_loader
import utils.metrics as metrics
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

def train():
    # 确保判别器输入维度与生成器输出一致
    assert opt.disc_input_size == opt.input_size, \
        "判别器输入维度(disc_input_size)必须等于生成器输出维度(input_size)"

    if opt.useGAN:
        # 生成器配置（输出时序数据）
        generator = BiLSTM(
            input_size=opt.input_size,
            rnn_size=opt.rnn_size,
            rnn_layers=opt.rnn_layers,
            dropout=opt.dropout,
            is_discriminator=False
        )

        # 判别器配置（输入为生成器输出）
        discriminator = BiLSTM(
            input_size=opt.disc_input_size,
            rnn_size=opt.disc_rnn_size,
            rnn_layers=opt.disc_rnn_layers,
            dropout=opt.disc_dropout,
            is_discriminator=True
        )

        trainer = Trainer(generator, discriminator)
    else:
        # 普通模式使用标准配置
        model = BiLSTM(
            input_size=opt.input_size,
            rnn_size=opt.rnn_size,
            rnn_layers=opt.rnn_layers,
            dropout=opt.dropout
        )
        trainer = Trainer(model)

    trainer.start_train()


def predict():
    device = opt.device

    # 初始化生成器模型
    generator = BiLSTM(
        input_size=opt.input_size,
        rnn_size=opt.rnn_size,
        rnn_layers=opt.rnn_layers,
        dropout=opt.dropout,
        is_discriminator=False
    ).to(device)
    generator.eval()

    # 加载checkpoint（兼容GAN模式和普通模式）
    checkpoint = torch.load(opt.resume_path, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
    else:
        generator.load_state_dict(checkpoint['state_dict'])

    # 数据加载
    loader = get_predict_loader(
        opt.data_dir,
        step=opt.time_tri,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers
    )

    # 预测过程
    y_predict, y_true = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Predicting"):
            X = X.to(device)
            pred = generator(X).cpu()
            y_predict.append(pred)
            y_true.append(y)

    # 拼接结果
    y_predict = torch.cat(y_predict, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    # 将每个样本的时序输出展平为 (step*7) 维向量
    y_predict_flat = y_predict.reshape(y_predict.shape[0], -1)
    y_true_flat = y_true.reshape(y_true.shape[0], -1)

    # 计算指标
    met = metrics.scores(y_true_flat, y_predict_flat)
    print(f"Test Metrics: {met}")



    # 保存结果
    # 读取原始数据
    df = pd.read_csv(opt.data_dir)

    # 根据 transform_dataset 计算预测对应的行：从 2*step - 1 行开始，总行数应为 (N - 2*step + 1)
    step = opt.time_tri  # 假设此处 step 与 opt.time_tri 一致
    df = df.iloc[2 * step - 1:].reset_index(drop=True)

    # 检查一下 df 的长度是否和预测样本数匹配
    # last_step_pred 的形状应为 (num_samples, 7)
    last_step_pred = y_predict[:, -1, :]  # 如果你希望保存预测窗口最后一时刻的全部7个特征

    # 此时，df 的行数应与 last_step_pred.shape[0] 一致
    if df.shape[0] != last_step_pred.shape[0]:
        print(f"警告：原始数据行数 {df.shape[0]} 与预测样本数 {last_step_pred.shape[0]} 不匹配！")

    # 将每个预测维度分别保存到新的列中（例如 pred_feature_0 到 pred_feature_6）
    for i in range(7):
        df[f"pred_feature_{i}"] = last_step_pred[:, i]

    df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        predict()