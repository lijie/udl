"""
demo1_numpy.py — 用纯 NumPy 手写神经网络拟合复合函数

目标函数 F(x):
    F(x) = beta3 + omega3 * cos(
                beta2 + omega2 * exp(
                    beta1 + omega1 * sin(
                        beta0 + omega0 * x
                    )
                )
            )

学习要点:
  - 手动实现前向传播 (forward pass)
  - 手动实现反向传播 (backpropagation)
  - SGD + Momentum 优化器
  - 训练/测试数据可视化

运行方式:
    uv run python demos/demo1_numpy.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_font_utils import configure_chinese_font


PLOT_FONT = configure_chinese_font()

# ─────────────────────────────────────────────
# 0. 超参数 / 全局配置
# ─────────────────────────────────────────────
SEED = 42           # 随机种子，保证可复现
X_MIN, X_MAX = -3.0, 3.0   # x 的采样范围
N_TRAIN = 200       # 训练样本数
N_TEST  = 100       # 测试样本数
NOISE_STD = 0.05    # 采样时加入的高斯噪声标准差

# 神经网络结构
N_HIDDEN_LAYERS = 4   # 隐藏层数
D_HIDDEN = 64         # 每层隐藏单元数

# 训练超参
EPOCHS    = 3000
LR        = 0.01   # 学习率
MOMENTUM  = 0.9    # SGD Momentum 系数
LOG_EVERY = 100    # 每隔多少 epoch 打印一次日志


# ─────────────────────────────────────────────
# 1. 目标函数
# ─────────────────────────────────────────────

def make_target_function(seed=SEED):
    """
    随机生成 beta / omega 参数，返回:
      - params: dict，保存所有参数值（方便打印复现）
      - F: callable, F(x) -> y

    函数结构 (由内到外):
        z0 = beta0 + omega0 * x
        z1 = beta1 + omega1 * sin(z0)
        z2 = beta2 + omega2 * exp(z1)
        y  = beta3 + omega3 * cos(z2)
    """
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
    """
    在 [X_MIN, X_MAX] 上随机采 n 个点，加入高斯噪声。
    返回 (x, y_noisy)，形状均为 (n, 1)。
    """
    x = rng.uniform(X_MIN, X_MAX, size=(n, 1))
    y_clean = F(x)
    y_noisy = y_clean + rng.normal(0, noise_std, size=y_clean.shape)
    return x, y_noisy


# ─────────────────────────────────────────────
# 2. 神经网络（纯 NumPy）
# ─────────────────────────────────────────────

class NumpyNet:
    """
    全连接深度神经网络，使用 Tanh 激活函数。

    网络结构:
        Linear → Tanh → Linear → Tanh → ... → Linear (输出层, 无激活)

    参数:
        layer_dims: list[int], 每层的维度，例如 [1, 64, 64, 64, 64, 1]
    """

    def __init__(self, layer_dims: list[int], seed=SEED):
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1   # 权重矩阵的数量

        # 用 He 初始化（适合 tanh 也够用）
        rng = np.random.default_rng(seed + 1)
        self.weights = []   # W[i]: shape (layer_dims[i+1], layer_dims[i])
        self.biases  = []   # b[i]: shape (layer_dims[i+1], 1)
        for i in range(self.n_layers):
            fan_in = layer_dims[i]
            scale = np.sqrt(1.0 / fan_in)   # Xavier 初始化
            W = rng.normal(0, scale, (layer_dims[i+1], layer_dims[i]))
            b = np.zeros((layer_dims[i+1], 1))
            self.weights.append(W)
            self.biases.append(b)

        # SGD + Momentum 的速度项（与权重/偏置同形状）
        self.vW = [np.zeros_like(W) for W in self.weights]
        self.vb = [np.zeros_like(b) for b in self.biases]

    # ── 前向传播 ──────────────────────────────

    def forward(self, x):
        """
        x: (batch, 1) —— 一列输入
        返回预测值 y_hat: (batch, 1)，同时缓存中间结果用于反向传播。

        缓存结构:
            self._cache = list of (A_prev, Z)
            A_prev: 该层输入（前一层激活值）
            Z:      线性变换结果 W @ A_prev + b（未经激活）
        """
        self._cache = []
        A = x.T   # 转为 (features, batch) 方便矩阵乘法

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            A_prev = A
            Z = W @ A_prev + b          # 线性变换: (out, batch)
            is_last_layer = (i == self.n_layers - 1)
            if is_last_layer:
                A = Z                   # 输出层：恒等激活（线性回归）
            else:
                A = np.tanh(Z)          # 隐藏层：Tanh 激活
            self._cache.append((A_prev, Z))

        return A.T   # 转回 (batch, 1)

    # ── 损失函数 ──────────────────────────────

    @staticmethod
    def mse_loss(y_hat, y_true):
        """均方误差: mean( (y_hat - y_true)^2 )"""
        diff = y_hat - y_true
        return np.mean(diff ** 2)

    # ── 反向传播 ──────────────────────────────

    def backward(self, y_hat, y_true):
        """
        根据 MSE 损失计算每层权重/偏置的梯度。

        反向传播核心公式（以第 i 层为例）：
            dL/dZ_i = dL/dA_i * tanh'(Z_i)   （隐藏层）
            dL/dZ_i = dL/dA_i                  （输出层）
            dL/dW_i = dZ_i @ A_{i-1}.T / batch
            dL/db_i = mean(dZ_i, axis=1, keepdims=True)
            dL/dA_{i-1} = W_i.T @ dZ_i          （传给下一层）

        返回: (grads_W, grads_b) 两个列表
        """
        batch = y_hat.shape[0]

        # MSE 对输出的梯度: dL/dy_hat = 2*(y_hat - y_true) / batch
        # 因为 mean = sum/batch，链式法则带出 1/batch
        dA = (2.0 / batch) * (y_hat - y_true)   # (batch, 1)
        dA = dA.T   # (1, batch)

        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        for i in reversed(range(self.n_layers)):
            A_prev, Z = self._cache[i]
            is_last_layer = (i == self.n_layers - 1)

            # 1. 激活函数的梯度
            if is_last_layer:
                dZ = dA   # 输出层无激活，梯度直接传过来
            else:
                # tanh'(z) = 1 - tanh(z)^2
                dZ = dA * (1.0 - np.tanh(Z) ** 2)

            # 2. 对权重和偏置的梯度
            grads_W[i] = dZ @ A_prev.T          # (out, in)
            grads_b[i] = np.mean(dZ, axis=1, keepdims=True)  # (out, 1)

            # 3. 将梯度向前传播
            dA = self.weights[i].T @ dZ          # (in, batch)

        return grads_W, grads_b

    # ── SGD + Momentum 更新 ────────────────────

    def update(self, grads_W, grads_b, lr=LR, momentum=MOMENTUM):
        """
        Momentum SGD 更新规则:
            v = momentum * v - lr * grad
            param += v

        Momentum 的作用：累积历史梯度方向，抑制振荡，加速收敛。
        """
        for i in range(self.n_layers):
            self.vW[i] = momentum * self.vW[i] - lr * grads_W[i]
            self.vb[i] = momentum * self.vb[i] - lr * grads_b[i]
            self.weights[i] += self.vW[i]
            self.biases[i]  += self.vb[i]

    def predict(self, x):
        """推理接口，直接返回 forward 结果。"""
        return self.forward(x)

    def grad_norm(self, grads_W, grads_b):
        """计算所有梯度的 L2 范数，用于监控训练稳定性。"""
        total = sum(np.sum(g**2) for g in grads_W + grads_b)
        return np.sqrt(total)


# ─────────────────────────────────────────────
# 3. 训练循环
# ─────────────────────────────────────────────

def train(net, x_train, y_train, x_test, y_test,
          epochs=EPOCHS, lr=LR, momentum=MOMENTUM, log_every=LOG_EVERY):
    """
    训练 net，返回 (train_losses, test_losses) 两个 list。

    每 log_every 步打印:
        Epoch | Train Loss | Test Loss | Gradient Norm
    """
    train_losses = []
    test_losses  = []

    print(f"\n{'─'*60}")
    print(f"  开始训练 | epochs={epochs}, lr={lr}, momentum={momentum}")
    print(f"  网络结构: {net.layer_dims}")
    print(f"{'─'*60}")
    print(f"{'Epoch':>8}  {'TrainLoss':>12}  {'TestLoss':>12}  {'GradNorm':>12}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        # ── 前向 ──────────────────────
        y_hat_train = net.forward(x_train)
        loss_train  = NumpyNet.mse_loss(y_hat_train, y_train)

        # ── 反向 ──────────────────────
        grads_W, grads_b = net.backward(y_hat_train, y_train)

        # ── 更新 ──────────────────────
        net.update(grads_W, grads_b, lr=lr, momentum=momentum)

        # ── 评估测试集（不参与梯度计算）──
        y_hat_test = net.predict(x_test)
        loss_test  = NumpyNet.mse_loss(y_hat_test, y_test)

        train_losses.append(float(loss_train))
        test_losses.append(float(loss_test))

        if epoch % log_every == 0 or epoch == 1:
            gnorm = net.grad_norm(grads_W, grads_b)
            print(f"{epoch:>8}  {loss_train:>12.6f}  {loss_test:>12.6f}  {gnorm:>12.6f}")

    print(f"{'─'*60}")
    print(f"  训练完成! 最终 Train Loss={train_losses[-1]:.6f}, Test Loss={test_losses[-1]:.6f}")
    print(f"{'─'*60}\n")

    return train_losses, test_losses


# ─────────────────────────────────────────────
# 4. 可视化
# ─────────────────────────────────────────────

def plot_results(F, x_train, y_train, x_test, y_test,
                 net, train_losses, test_losses, params):
    """
    在一张图上展示:
      左图: 原函数曲线 / 训练数据 / 测试数据 / 网络预测曲线
      右图: 训练/测试 Loss 曲线
    """
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Demo1 (NumPy) — 神经网络拟合复合函数", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── 左图: 拟合效果 ─────────────────────────
    ax1 = fig.add_subplot(gs[0])

    # 稠密 x 网格，用于绘制连续曲线
    x_dense = np.linspace(X_MIN, X_MAX, 500).reshape(-1, 1)
    y_true  = F(x_dense)

    # 网络在稠密网格上的预测
    y_pred = net.predict(x_dense)

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

    # 在图上标注最终 Loss
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

    plt.savefig("demos/demo1_numpy_result.png", dpi=150, bbox_inches='tight')
    print("图像已保存至 demos/demo1_numpy_result.png")
    plt.show()


# ─────────────────────────────────────────────
# 5. 主程序
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Demo1 — NumPy 手写神经网络拟合复合函数")
    if PLOT_FONT is not None:
        print(f"  Matplotlib 字体: {PLOT_FONT}")
    print("=" * 60)

    # ── 生成目标函数 ──────────────────────────
    params, F = make_target_function(seed=SEED)
    print("\n目标函数参数 (随机生成):")
    for k, v in params.items():
        print(f"    {k:8s} = {v:+.6f}")

    # ── 采样数据 ──────────────────────────────
    rng = np.random.default_rng(SEED + 100)
    x_train, y_train = sample_data(F, N_TRAIN, rng)
    x_test,  y_test  = sample_data(F, N_TEST,  rng)
    print(f"\n训练集: {x_train.shape[0]} 样本 | 测试集: {x_test.shape[0]} 样本")
    print(f"x 范围: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"y 范围 (train): [{y_train.min():.2f}, {y_train.max():.2f}]")

    # ── 构造网络 ──────────────────────────────
    # 输入维度=1, 若干隐藏层, 输出维度=1
    layer_dims = [1] + [D_HIDDEN] * N_HIDDEN_LAYERS + [1]
    net = NumpyNet(layer_dims, seed=SEED)
    total_params = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
    print(f"\n网络结构: {layer_dims}")
    print(f"总参数量: {total_params}")

    # ── 训练 ──────────────────────────────────
    train_losses, test_losses = train(
        net, x_train, y_train, x_test, y_test,
        epochs=EPOCHS, lr=LR, momentum=MOMENTUM, log_every=LOG_EVERY
    )

    # ── 可视化 ────────────────────────────────
    plot_results(F, x_train, y_train, x_test, y_test,
                 net, train_losses, test_losses, params)


if __name__ == "__main__":
    main()
