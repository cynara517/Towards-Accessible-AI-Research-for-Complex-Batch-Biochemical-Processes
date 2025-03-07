#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn.functional as F
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from models.lstm_DP import BiLSTM
from models.lossFun_DP import TimeSeriesGANLoss
from utils.loader import get_loader, get_predict_loader
import utils.metrics as metrics
import opt_DP as opt
import pytorch_warmup as warmup
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class Trainer:
    def __init__(self, generator, discriminator=None):
        self.device = opt.device
        self.useGAN = opt.useGAN
        # 参数存在性校验
        required_params = ['gen_lr', 'disc_lr', 'disc_dp']
        for param in required_params:
            if not hasattr(opt, param):
                raise ValueError(f"opt_DP.py缺少必需参数: {param}")
        # 模型初始化
        if self.useGAN:
            self.generator = generator.to(self.device)
            self.discriminator = discriminator.to(self.device)
            self.model = None
        else:
            self.model = generator.to(self.device)
            self.generator = self.discriminator = None

        # 数据加载
        self.train_loader, self.val_loader,_ = get_loader(
            opt.data_dir,
            batch_size=opt.batch_size,
            step=opt.time_tri,
            num_workers=opt.num_workers
        )

        # 优化器配置
        if self.useGAN:
            self.optimizer_G = torch.optim.AdamW(
                self.generator.parameters(),
                lr=opt.gen_lr,
                weight_decay=opt.weight_decay
            )
            self.optimizer_D = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=opt.disc_lr,
                weight_decay=opt.disc_weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

        # 损失函数
        self.loss_fn = TimeSeriesGANLoss(
            alpha=opt.alpha,
            gamma=opt.gamma,
            recon_weight=opt.recon_weight,
            adv_weight=opt.adv_weight,
            cosine_weight=opt.cosine_weight,
            use_focal=opt.use_focal
        ) if self.useGAN else nn.MSELoss()

        # 差分隐私引擎
        self.privacy_engine = None
        if opt.dp:
            self._init_dp()

        # 学习率调度
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, T_max=opt.epochs
        ) if self.useGAN else None
        self.warmup_scheduler = warmup.LinearWarmup(
            self.optimizer_G, warmup_period=5
        ) if self.useGAN else None

        # 训练记录
        self.best_loss = float('inf')
        os.makedirs(opt.checkpoint_dir, exist_ok=True)

    def _get_max_grad_norm(self, model):
        """动态生成分层梯度裁剪参数（类方法）"""
        param_groups = []
        print("\n===== 模型参数结构 =====")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"参数名: {name} | 形状: {param.shape}")
                if 'lstm' in name.lower():
                    param_groups.append(1.2)  # LSTM层阈值
                elif 'linear' in name.lower():
                    param_groups.append(0.8)  # 全连接层阈值
                else:
                    param_groups.append(1.0)  # 默认阈值
        print(f"总参数层数: {len(param_groups)}")
        return param_groups

    def _init_dp(self):
        if self.useGAN:
            self.privacy_engine = PrivacyEngine()

            # 修改调用方式为类方法
            gen_max_grad_norm = self._get_max_grad_norm(self.generator)

            self.generator, self.optimizer_G, self.train_loader = \
                self.privacy_engine.make_private(
                    module=self.generator,
                    optimizer=self.optimizer_G,
                    data_loader=self.train_loader,
                    noise_multiplier=opt.noise_multiplier,
                    max_grad_norm=gen_max_grad_norm,  # 使用动态生成的参数
                    clipping="per_layer"
                )
        else:
            # 非GAN模式
            self.privacy_engine = PrivacyEngine(secure_mode=opt.secure_rng)
            self.model, self.optimizer, self.train_loader = \
                self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=opt.noise_multiplier,
                    max_grad_norm=opt.max_grad_norm,
                    clipping="per_layer"
                )

    def train_epoch(self, epoch):
        self.generator.train() if self.useGAN else self.model.train()
        total_loss = 0.0

        with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=32,
                optimizer=self.optimizer_G if self.useGAN else self.optimizer  # 非 GAN 模式下只传递一个优化器
        ) as memory_safe_loader:
            # 以下代码保持不变
            pbar = tqdm(memory_safe_loader, desc=f"Epoch {epoch}")
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)
                fake = self.generator(X)

                if self.useGAN:
                    # 以下 GAN 模式的代码保持不变
                    self.optimizer_D.zero_grad()
                    real_pred = self.discriminator(y)
                    fake_pred = self.discriminator(fake.detach())
                    loss_D = self.loss_fn(
                        real_data=y,
                        disc_real_output=real_pred,
                        disc_fake_output=fake_pred,
                        mode='discriminator'
                    )
                    loss_D.backward()
                    self.optimizer_D.step()  # 必须立即更新

                    self.optimizer_G.zero_grad()
                    fake_pred = self.discriminator(fake)
                    loss_G = self.loss_fn(
                        real_data=y,
                        disc_real_output=real_pred.detach(),
                        disc_fake_output=fake_pred,
                        mode='generator',
                        fake_data=fake
                    )
                    loss_G.backward()
                    self.optimizer_G.step()  # 必须立即更新

                    total_loss += loss_G.item()
                    pbar.set_postfix({'Loss_G': loss_G.item(), 'Loss_D': loss_D.item()})
                else:
                    # 以下普通模式的代码保持不变
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.optimizer.step()  # 必须立即更新

                    total_loss += loss.item()
                    pbar.set_postfix({'Loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        model = self.generator if self.useGAN else self.model
        model.eval()
        total_loss = 0.0
        results = {'true': [], 'pred': []}

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                # 对于 GAN 模式，验证时只计算 MSE 重构误差
                loss = F.mse_loss(pred, y) if self.useGAN else self.loss_fn(pred, y)
                total_loss += loss.item()

                results['true'].append(y.cpu())
                results['pred'].append(pred.cpu())

        y_true = torch.cat(results['true'], dim=0).numpy()
        y_pred = torch.cat(results['pred'], dim=0).numpy()
        y_true = y_true[:, -1, :]
        y_pred = y_pred[:, -1, :]
        metrics_dict = metrics.scores(y_true, y_pred)
        return total_loss / len(self.val_loader), metrics_dict

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict() if self.useGAN else None,
            'discriminator_state_dict': self.discriminator.state_dict() if self.useGAN else None,
            'model_state_dict': None if self.useGAN else self.model.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict() if self.useGAN else None,
            'optimizer_D': self.optimizer_D.state_dict() if self.useGAN else None,
            'loss': self.best_loss,
        }
        filename = os.path.join(opt.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(state, filename)
        if is_best:
            best_path = os.path.join(opt.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)

    def start_train(self):
        # 在训练开始前添加引擎状态检查
        if opt.dp and self.useGAN:
            assert hasattr(self.privacy_engine, "accountant"), "隐私会计未初始化"
            print("隐私引擎配置:")
            print(f" - Secure RNG: {self.privacy_engine.secure_mode}")
            print(f" - 生成器噪声系数: {opt.noise_multiplier}")
            if opt.disc_dp:
                print(f" - 判别器噪声系数: {opt.disc_noise_multiplier}")

        # ... 后续训练代码保持不变 ...
        for epoch in range(opt.epochs):
            # ... [原有训练代码] ...
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate()

            # 更新学习率
            if self.useGAN:
                self.scheduler_G.step()
                self.warmup_scheduler.dampen()

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        if opt.useGAN:
            generator = BiLSTM(
                input_size=opt.input_size,
                rnn_size=opt.rnn_size,
                rnn_layers=opt.rnn_layers,
                dropout=opt.dropout,
                is_discriminator=False
            )
            discriminator = BiLSTM(
                input_size=opt.disc_input_size,
                rnn_size=opt.disc_rnn_size,
                rnn_layers=opt.disc_rnn_layers,
                dropout=opt.disc_dropout,
                is_discriminator=True
            )
            trainer = Trainer(generator, discriminator)
        else:
            model = BiLSTM(
                input_size=opt.input_size,
                rnn_size=opt.rnn_size,
                rnn_layers=opt.rnn_layers,
                dropout=opt.dropout
            )
            trainer = Trainer(model)
        trainer.start_train()
    else:
        # 预测代码（与之前保持一致）
        pass