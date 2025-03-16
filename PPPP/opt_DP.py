# opt_DP.py 升级版
import torch


# ========== 核心开关 ==========
useGAN = True           # 是否启用GAN模式
dp = True               # 是否启用差分隐私
# ========== 差分隐私 ==========
delta = 1e-5                  # 隐私预算参数
target_epsilon = 1.0          # 目标ε值

# ========== 模型架构 ==========
# ---- 生成器参数 ----
input_size = 7          # 输入特征维度（必须与数据一致）
rnn_size = 256          # LSTM隐藏层大小
rnn_layers = 2          # LSTM层数
dropout = 0.3           # 生成器Dropout

# ---- 判别器参数 ----
disc_input_size = 7     # 输入维度（需等于生成器输出维度）
disc_rnn_size = 128     # 判别器隐藏层大小
disc_rnn_layers = 2     # 判别器层数
disc_dropout = 0.3      # 判别器Dropout

# ========== GAN训练参数 ==========
gan_lambda = 0.1        # 废弃参数
recon_weight = 1.0      # 重构损失权重
adv_weight = 0.1        # 对抗损失权重
cosine_weight = 0.2     # 余弦相似度权重
use_focal = True        # 是否使用Focal Loss
alpha = 0.75            # Focal Loss参数
gamma = 2               # Focal Loss参数
weight_decay=1e-2
# ========== 差分隐私 ==========
noise_multiplier = 1.3  # 生成器噪声系数
# ===== 原错误参数 =====
# max_grad_norm = 1.0  # 错误：单个浮点数

# ===== 修正为分层参数 =====

disc_dp = True          # 判别器是否启用DP（新増）
# ===== 新增判别器专用参数 =====
disc_noise_multiplier = 1.5


# ========== 优化器参数 ==========
LEARNING_RATE = 1e-3    # 废弃
gen_lr = 1e-4           # 生成器学习率（新増）
disc_lr = 2e-4          # 判别器学习率（新増）
WEIGHT_DECAY = 1e-2     # 生成器权重衰减
disc_weight_decay = 1e-3# 判别器权重衰减（新増）

# ========== 训练配置 ==========
batch_size = 128
epochs = 50
num_workers = 2      # 建议改为num_workers
accumulation_step = 1   # 梯度累加步数
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = "adamw"     # 优化器类型
lrsc = "warmup"         # 学习率调度策略
lr=1e-4
# ========== 损失函数 ==========
loss = "TimeSeriesGAN" if useGAN else "MSE"  # 自动匹配损失类型

# ========== 数据参数 ==========
time_tri = 128           # 输入序列长度
testRatio = 0.2        # 测试集比例
valRatio = 0.2          # 验证集比例
data_dir = "./data/EP#1.csv"
save_dir = "./evaluate/"
filename = "dataname"   # 数据集标识

# ========== 模型存储 ==========
keep = True             # 是否保存模型
resume = False          # 是否加载checkpoint
resume_path = "./weights/best_model.pth"  
checkpoint_dir = "./weights"
useGAN_weights = False  
GAN_path = "./weights/ganmodel.pkl"     
NAME = "GANBILSTM"     
