"""
numpy_classification.py — 用纯 NumPy 训练二分类神经网络拟合经典二维数据集

覆盖数据集:
  - moons: 月牙形
  - circles: 环形
  - spiral: 螺旋线

运行方式:
    uv run python demos/demo2/numpy_classification.py
    uv run python demos/demo2/numpy_classification.py --dataset moons
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
COMMON_DIR = PROJECT_ROOT / "demos" / "common"
ARTIFACTS_DIR = CURRENT_DIR / "artifacts"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from mpl_font_utils import configure_chinese_font
from toy_classification_data import build_demo2_datasets, transform_points


PLOT_FONT = configure_chinese_font()

SEED = 42
N_TRAIN = 480
N_TEST = 240

N_HIDDEN_LAYERS = 3
D_HIDDEN = 64

EPOCHS = 2500
LR = 0.08
MOMENTUM = 0.9
LOG_EVERY = 125

AVAILABLE_DATASETS = ["moons", "circles", "spiral"]


class NumpyBinaryClassifier:
    """A fully-connected binary classifier with Tanh hidden layers."""

    def __init__(self, layer_dims: list[int], seed: int = SEED):
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1

        rng = np.random.default_rng(seed + 7)
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = layer_dims[i]
            scale = np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, (layer_dims[i + 1], fan_in)))
            self.biases.append(np.zeros((layer_dims[i + 1], 1)))

        self.vW = [np.zeros_like(weight) for weight in self.weights]
        self.vb = [np.zeros_like(bias) for bias in self.biases]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        activation = x.T

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            prev_activation = activation
            linear = weight @ prev_activation + bias
            is_last = index == self.n_layers - 1
            if is_last:
                activation = 1.0 / (1.0 + np.exp(-linear))
            else:
                activation = np.tanh(linear)
            self._cache.append((prev_activation, linear))

        return activation.T

    @staticmethod
    def bce_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        loss = -(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
        return float(np.mean(loss))

    @staticmethod
    def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
        preds = (probs >= 0.5).astype(np.float64)
        return float(np.mean(preds == y_true))

    def backward(self, probs: np.ndarray, y_true: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch = probs.shape[0]
        delta = (probs - y_true).T

        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        for index in reversed(range(self.n_layers)):
            prev_activation, linear = self._cache[index]
            grads_W[index] = (delta @ prev_activation.T) / batch
            grads_b[index] = np.mean(delta, axis=1, keepdims=True)

            if index > 0:
                delta = (self.weights[index].T @ delta) * (1.0 - np.tanh(self._cache[index - 1][1]) ** 2)

        return grads_W, grads_b

    def update(self, grads_W: list[np.ndarray], grads_b: list[np.ndarray], lr: float, momentum: float) -> None:
        for index in range(self.n_layers):
            self.vW[index] = momentum * self.vW[index] - lr * grads_W[index]
            self.vb[index] = momentum * self.vb[index] - lr * grads_b[index]
            self.weights[index] += self.vW[index]
            self.biases[index] += self.vb[index]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def grad_norm(self, grads_W: list[np.ndarray], grads_b: list[np.ndarray]) -> float:
        return float(np.sqrt(sum(np.sum(grad ** 2) for grad in grads_W + grads_b)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demo2 NumPy classification demos.")
    parser.add_argument(
        "--dataset",
        choices=["all", *AVAILABLE_DATASETS],
        default="all",
        help="Which dataset to run.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Override the number of epochs.")
    parser.add_argument("--no-show", action="store_true", help="Save figures without calling plt.show().")
    return parser.parse_args()


def train_one_dataset(
    dataset,
    epochs: int,
    lr: float,
    momentum: float,
    log_every: int,
    seed: int,
) -> dict[str, object]:
    layer_dims = [2] + [D_HIDDEN] * N_HIDDEN_LAYERS + [1]
    net = NumpyBinaryClassifier(layer_dims, seed=seed)

    train_losses: list[float] = []
    test_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []

    print(f"\n{'=' * 72}")
    print(f"[NumPy] 数据集: {dataset.title}")
    print(f"训练样本: {len(dataset.train_x)} | 测试样本: {len(dataset.test_x)}")
    print(f"标准化前 train x 范围: [{dataset.train_raw.min():+.3f}, {dataset.train_raw.max():+.3f}]")
    print(f"标准化均值: {dataset.mean.ravel()} | 标准差: {dataset.std.ravel()}")
    print(f"网络结构: {layer_dims}")
    print(f"{'─' * 72}")
    print(f"{'Epoch':>8}  {'TrainLoss':>12}  {'TestLoss':>12}  {'TrainAcc':>10}  {'TestAcc':>10}  {'GradNorm':>10}")
    print(f"{'─' * 72}")

    for epoch in range(1, epochs + 1):
        train_probs = net.forward(dataset.train_x)
        train_loss = net.bce_loss(train_probs, dataset.y_train)
        train_acc = net.accuracy(train_probs, dataset.y_train)

        grads_W, grads_b = net.backward(train_probs, dataset.y_train)
        net.update(grads_W, grads_b, lr=lr, momentum=momentum)

        test_probs = net.predict_proba(dataset.test_x)
        test_loss = net.bce_loss(test_probs, dataset.y_test)
        test_acc = net.accuracy(test_probs, dataset.y_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if epoch % log_every == 0 or epoch == 1:
            grad_norm = net.grad_norm(grads_W, grads_b)
            print(
                f"{epoch:>8}  {train_loss:>12.6f}  {test_loss:>12.6f}"
                f"  {train_acc:>10.4f}  {test_acc:>10.4f}  {grad_norm:>10.4f}"
            )

    print(f"{'─' * 72}")
    print(
        "最终结果: "
        f"Train Loss={train_losses[-1]:.6f}, Test Loss={test_losses[-1]:.6f}, "
        f"Train Acc={train_accs[-1]:.4f}, Test Acc={test_accs[-1]:.4f}"
    )

    return {
        "dataset": dataset,
        "net": net,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
    }


def plot_results(results: list[dict[str, object]], output_path: Path, show_plot: bool) -> None:
    rows = len(results)
    fig = plt.figure(figsize=(16, 5 * rows))
    fig.suptitle("Demo2 (NumPy) — 经典二维分类数据集", fontsize=16, fontweight="bold")
    grid = gridspec.GridSpec(rows, 2, figure=fig, hspace=0.3, wspace=0.25)

    class_colors = {0: "steelblue", 1: "crimson"}

    for row, result in enumerate(results):
        dataset = result["dataset"]
        net = result["net"]
        train_losses = result["train_losses"]
        test_losses = result["test_losses"]
        train_accs = result["train_accs"]
        test_accs = result["test_accs"]

        ax_boundary = fig.add_subplot(grid[row, 0])
        x_min = min(dataset.train_raw[:, 0].min(), dataset.test_raw[:, 0].min()) - 0.4
        x_max = max(dataset.train_raw[:, 0].max(), dataset.test_raw[:, 0].max()) + 0.4
        y_min = min(dataset.train_raw[:, 1].min(), dataset.test_raw[:, 1].min()) - 0.4
        y_max = max(dataset.train_raw[:, 1].max(), dataset.test_raw[:, 1].max()) + 0.4
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
        mesh_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        mesh_points_scaled = transform_points(mesh_points, dataset.mean, dataset.std)
        probs = net.predict_proba(mesh_points_scaled).reshape(grid_x.shape)

        ax_boundary.contourf(grid_x, grid_y, probs, levels=24, cmap="RdBu_r", alpha=0.35)
        ax_boundary.contour(grid_x, grid_y, probs, levels=[0.5], colors="black", linewidths=1.2)

        for class_id in [0, 1]:
            train_mask = dataset.y_train.ravel() == class_id
            test_mask = dataset.y_test.ravel() == class_id
            ax_boundary.scatter(
                dataset.train_raw[train_mask, 0],
                dataset.train_raw[train_mask, 1],
                s=24,
                c=class_colors[class_id],
                alpha=0.75,
            )
            ax_boundary.scatter(
                dataset.test_raw[test_mask, 0],
                dataset.test_raw[test_mask, 1],
                s=44,
                c=class_colors[class_id],
                marker="^",
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
            )

        ax_boundary.set_title(
            f"{dataset.title}\nDecision Boundary | Test Acc={test_accs[-1]:.3f}"
        )
        ax_boundary.set_xlabel("x1")
        ax_boundary.set_ylabel("x2")
        ax_boundary.grid(True, alpha=0.2)
        ax_boundary.legend(
            handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", label="Train Class 0", markersize=7),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", label="Train Class 1", markersize=7),
                Line2D([0], [0], marker="^", color="black", markerfacecolor="white", label="Test Sample", markersize=7),
            ],
            loc="upper right",
            fontsize=8,
        )

        ax_loss = fig.add_subplot(grid[row, 1])
        epochs = np.arange(1, len(train_losses) + 1)
        ax_acc = ax_loss.twinx()

        loss_lines = ax_loss.plot(epochs, train_losses, color="orange", lw=1.8, label="Train BCE")
        loss_lines += ax_loss.plot(epochs, test_losses, color="green", lw=1.8, label="Test BCE")
        acc_lines = ax_acc.plot(epochs, train_accs, color="steelblue", lw=1.6, linestyle="--", label="Train Acc")
        acc_lines += ax_acc.plot(epochs, test_accs, color="crimson", lw=1.6, linestyle="--", label="Test Acc")

        ax_loss.set_title(f"{dataset.title}\nLoss / Accuracy Curves")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Binary Cross Entropy")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0.45, 1.02)
        ax_loss.grid(True, alpha=0.25)

        handles = loss_lines + acc_lines
        labels = [line.get_label() for line in handles]
        ax_loss.legend(handles, labels, loc="center right", fontsize=8)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n图像已保存至 {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    selected_names = None if args.dataset == "all" else [args.dataset]
    datasets = build_demo2_datasets(N_TRAIN, N_TEST, seed=SEED, selected_names=selected_names)

    print("=" * 72)
    print("  Demo2 — NumPy 经典二维数据集分类")
    if PLOT_FONT is not None:
        print(f"  Matplotlib 字体: {PLOT_FONT}")
    print(f"  数据集: {[dataset.name for dataset in datasets]}")
    print(f"  epochs={args.epochs}, lr={LR}, momentum={MOMENTUM}")
    print("=" * 72)

    results = []
    for index, dataset in enumerate(datasets):
        results.append(
            train_one_dataset(
                dataset,
                epochs=args.epochs,
                lr=LR,
                momentum=MOMENTUM,
                log_every=LOG_EVERY,
                seed=SEED + 100 * index,
            )
        )

    dataset_suffix = args.dataset if args.dataset != "all" else "all"
    output_path = ARTIFACTS_DIR / f"numpy_classification_{dataset_suffix}.png"
    plot_results(results, output_path=output_path, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
