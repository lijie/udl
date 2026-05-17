"""
demo1_pytorch.py — 用 PyTorch 拟合复合函数

目标函数 F(x) 与 demo1_numpy.py 完全相同，区别在于:
  - 使用 torch.nn.Sequential 构建网络
  - 自动微分取代手动反向传播
  - Adam 优化器（比 SGD+Momentum 收敛通常更稳定）
  - 与 numpy 版输出同样风格的可视化，方便对照

运行方式:
    uv run python demos/demo1_pytorch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mpl_font_utils import configure_chinese_font


PLOT_FONT = configure_chinese_font()

# ─────────────────────────────────────────────
# 0. 超参数
# ─────────────────────────────────────────────
SEED = 42
X_MIN, X_MAX = -3.0, 3.0
N_TRAIN = 200
N_TEST  = 100
NOISE_STD = 0.05

N_HIDDEN_LAYERS = 4
D_HIDDEN = 64

EPOCHS     = 3000
LR         = 1e-3    # Adam 默认学习率
BATCH_SIZE = 64      # 小批量梯度下降（numpy 版用全批量）
LOG_EVERY  = 100


# ─────────────────────────────────────────────
# 1. 目标函数（与 numpy 版完全相同）
# ─────────────────────────────────────────────

def make_target_function(seed=SEED):
    """随机生成参数并返回 (params, F)，逻辑见 demo1_numpy.py"""
    rng = np.random.default_rng(seed)
    lo, hi = -1.0, 1.0
    params = {
        'beta0':  rng.uniform(lo, hi),
        'omega0': rng.uniform(lo, hi),
        'beta1':  rng.uniform(lo, hi),
        'omega1': rng.uniform(lo, hi),
        'beta2':  rng.uniform(lo, hi),
        'omega2': rng.uniform(lo, hi),
        'beta3':  rng.uniform(lo, hi),
        'omega3': rng.uniform(lo, hi),
    }

    def F(x):
        z0 = params['beta0'] + params['omega0'] * x
        z1 = params['beta1'] + params['omega1'] * np.sin(z0)
        z2 = params['beta2'] + params['omega2'] * np.exp(z1)
        y  = params['beta3'] + params['omega3'] * np.cos(z2)
        return y

    return params, F


def sample_data(F, n, rng, noise_std=NOISE_STD):
    """采样带噪声数据，返回 (x, y) numpy arrays，形状 (n, 1)"""
    x = rng.uniform(X_MIN, X_MAX, size=(n, 1))
    y = F(x) + rng.normal(0, noise_std, size=(n, 1))
    return x, y


# ─────────────────────────────────────────────
# 2. PyTorch 网络定义
# ─────────────────────────────────────────────

def build_network(layer_dims: list[int]) -> nn.Sequential:
    """
    根据 layer_dims 构建全连接网络。

    例如 layer_dims = [1, 64, 64, 64, 64, 1] 生成:
        Linear(1→64) → Tanh
        Linear(64→64) → Tanh
        Linear(64→64) → Tanh
        Linear(64→64) → Tanh
        Linear(64→1)           ← 输出层无激活

    PyTorch 的 nn.Sequential 会自动:
      - 注册所有参数（供优化器和 autograd 使用）
      - 按顺序执行 forward
    """
    layers = []
    for i in range(len(layer_dims) - 1):
        in_dim  = layer_dims[i]
        out_dim = layer_dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        # 最后一层不加激活函数（线性输出，适合回归）
        if i < len(layer_dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def init_weights(model: nn.Sequential):
    """
    Xavier 均匀初始化权重，与 numpy 版保持一致风格。
    偏置初始化为 0。
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────
# 3. 训练循环
# ─────────────────────────────────────────────

def train(model, x_train_t, y_train_t, x_test_t, y_test_t,
          epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, log_every=LOG_EVERY):
    """
    使用 Adam 优化器训练模型，支持 mini-batch。

    Adam 相比 SGD+Momentum 的优势：
      - 自适应学习率（每个参数单独调整）
      - 一阶矩 + 二阶矩估计，收敛更快更稳

    返回 (train_losses, test_losses)，每个元素对应一个 epoch 的平均损失。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()   # torch 内置 MSE，等价于 mean((y_hat - y)^2)

    # DataLoader 自动打乱并分 batch
    dataset    = TensorDataset(x_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    test_losses  = []

    print(f"\n{'─'*60}")
    print(f"  开始训练 | epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print(f"  网络结构: {[m for m in model if isinstance(m, nn.Linear)]}")
    print(f"{'─'*60}")
    print(f"{'Epoch':>8}  {'TrainLoss':>12}  {'TestLoss':>12}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        model.train()  # 切换到训练模式（影响 Dropout/BN 等，本例实际无影响）
        epoch_loss = 0.0
        n_batches  = 0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()          # 每个 batch 前清零梯度
            y_hat = model(x_batch)         # 前向传播（PyTorch 自动追踪计算图）
            loss  = criterion(y_hat, y_batch)
            loss.backward()                # 反向传播（autograd 自动计算梯度）
            optimizer.step()               # Adam 参数更新
            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # 评估测试集（torch.no_grad 禁止记录梯度，节省显存/内存）
        model.eval()
        with torch.no_grad():
            y_hat_test = model(x_test_t)
            loss_test  = criterion(y_hat_test, y_test_t).item()
        test_losses.append(loss_test)

        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>8}  {avg_train_loss:>12.6f}  {loss_test:>12.6f}")

    print(f"{'─'*60}")
    print(f"  训练完成! 最终 Train Loss={train_losses[-1]:.6f}, Test Loss={test_losses[-1]:.6f}")
    print(f"{'─'*60}\n")

    return train_losses, test_losses


# ─────────────────────────────────────────────
# 4. 可视化（与 numpy 版保持一致风格）
# ─────────────────────────────────────────────

def plot_results(F, x_train, y_train, x_test, y_test,
                 model, train_losses, test_losses):
    """在单张图上展示拟合效果和 Loss 曲线"""
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Demo1 (PyTorch) — 神经网络拟合复合函数", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── 左图: 拟合效果 ─────────────────────────
    ax1 = fig.add_subplot(gs[0])

    x_dense = np.linspace(X_MIN, X_MAX, 500).reshape(-1, 1)
    y_true  = F(x_dense)

    # PyTorch 推理：需要转 Tensor，推理后转回 numpy
    model.eval()
    with torch.no_grad():
        x_dense_t = torch.tensor(x_dense, dtype=torch.float32)
        y_pred = model(x_dense_t).numpy()

    ax1.plot(x_dense, y_true,  color='steelblue',  lw=2.5, label='目标函数 F(x)', zorder=3)
    ax1.scatter(x_train, y_train, s=12, alpha=0.6, color='orange',
                label=f'训练数据 (n={len(x_train)})', zorder=2)
    ax1.scatter(x_test,  y_test,  s=12, alpha=0.6, color='green',
                label=f'测试数据  (n={len(x_test)})', zorder=2)
    ax1.plot(x_dense, y_pred,  color='crimson',   lw=2.0, linestyle='--',
             label='网络预测', zorder=4)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("函数拟合对比")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 右图: Loss 曲线 ────────────────────────
    ax2 = fig.add_subplot(gs[1])
    epochs_range = np.arange(1, len(train_losses) + 1)

    ax2.semilogy(epochs_range, train_losses, color='orange', lw=1.8, label='Train Loss')
    ax2.semilogy(epochs_range, test_losses,  color='green',  lw=1.8, label='Test Loss')

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss (log scale)")
    ax2.set_title("训练曲线")
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    ax2.annotate(f"Final Train: {train_losses[-1]:.5f}",
                 xy=(len(train_losses), train_losses[-1]),
                 xytext=(-120, 20), textcoords='offset points',
                 fontsize=8, color='orange',
                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.2))
    ax2.annotate(f"Final Test:  {test_losses[-1]:.5f}",
                 xy=(len(test_losses), test_losses[-1]),
                 xytext=(-120, -30), textcoords='offset points',
                 fontsize=8, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

    plt.savefig("demos/demo1_pytorch_result.png", dpi=150, bbox_inches='tight')
    print("图像已保存至 demos/demo1_pytorch_result.png")
    plt.show()


# ─────────────────────────────────────────────
# 5. 主程序
# ─────────────────────────────────────────────

def main():
    # 固定所有随机种子，确保可复现
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  Demo1 — PyTorch 神经网络拟合复合函数")
    if PLOT_FONT is not None:
        print(f"  Matplotlib 字体: {PLOT_FONT}")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  设备: CPU")
    print("=" * 60)

    # ── 生成目标函数 ──────────────────────────
    params, F = make_target_function(seed=SEED)
    print("\n目标函数参数 (随机生成):")
    for k, v in params.items():
        print(f"    {k:8s} = {v:+.6f}")

    # ── 采样数据 ──────────────────────────────
    rng = np.random.default_rng(SEED + 100)
    x_train_np, y_train_np = sample_data(F, N_TRAIN, rng)
    x_test_np,  y_test_np  = sample_data(F, N_TEST,  rng)
    print(f"\n训练集: {N_TRAIN} 样本 | 测试集: {N_TEST} 样本")

    # 转为 PyTorch Tensor（float32 是神经网络的标准精度）
    x_train_t = torch.tensor(x_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    x_test_t  = torch.tensor(x_test_np,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np,  dtype=torch.float32)

    # ── 构造网络 ──────────────────────────────
    layer_dims = [1] + [D_HIDDEN] * N_HIDDEN_LAYERS + [1]
    model = build_network(layer_dims)
    init_weights(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n网络结构:\n{model}")
    print(f"\n总参数量: {total_params}")

    # ── 训练 ──────────────────────────────────
    train_losses, test_losses = train(
        model, x_train_t, y_train_t, x_test_t, y_test_t,
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, log_every=LOG_EVERY
    )

    # ── 可视化 ────────────────────────────────
    plot_results(F, x_train_np, y_train_np, x_test_np, y_test_np,
                 model, train_losses, test_losses)


if __name__ == "__main__":
    main()
