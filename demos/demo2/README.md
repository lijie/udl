# demo2

`demo2` 用三类经典二维分类数据集来观察深度神经网络的拟合能力和调参表现：

- `moons`: 月牙形数据集
- `circles`: 环形数据集
- `spiral`: 螺旋线数据集

目录内保留两套实现：

- `numpy_classification.py`: 纯 NumPy 版本，手写前向传播、反向传播和 `SGD + Momentum`。
- `pytorch_classification.py`: PyTorch CPU 版本，使用 `nn.Sequential` 和 `Adam`。
- `run.py`: 统一运行入口，可以切换实现和数据集。

## 目录结构

```text
demos/demo2/
├── README.md
├── run.py
├── numpy_classification.py
├── pytorch_classification.py
└── artifacts/
```

## 运行方式

同步环境：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv sync --python 3.12 --frozen
```

查看可运行内容：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo2/run.py --list
```

运行全部数据集的 NumPy 版本：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo2/run.py numpy --dataset all
```

只运行 `spiral` 的 PyTorch 版本：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo2/run.py pytorch --dataset spiral
```

如果只想快速验证流程，可以缩短训练并关闭图窗：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo2/run.py all --dataset moons --epochs 200 --no-show
```

## 输出

- NumPy 图像: `demos/demo2/artifacts/numpy_classification_<dataset>.png`
- PyTorch 图像: `demos/demo2/artifacts/pytorch_classification_<dataset>.png`

当 `--dataset all` 时，会在同一张总览图中同时展示三类数据集的决策边界、训练损失和精度曲线。
