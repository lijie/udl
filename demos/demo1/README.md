# demo1

`demo1` 用一个随机生成的复合函数做回归拟合，保留两套实现：

- `numpy_fit.py`: 纯 NumPy 手写前向传播、反向传播和 SGD + Momentum。
- `pytorch_fit.py`: PyTorch CPU 版本，使用 `nn.Sequential` + Adam。
- `run.py`: 统一运行入口，方便后续继续给 `demo1` 增加更多变体。

## 目录结构

```text
demos/demo1/
├── README.md
├── run.py
├── numpy_fit.py
├── pytorch_fit.py
└── artifacts/
```

`artifacts/` 下保存运行时生成的图像结果，默认已加入忽略列表。

## 运行方式

先同步环境：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv sync --python 3.12 --frozen
```

列出可运行实现：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo1/run.py --list
```

运行 NumPy 版本：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo1/run.py numpy
```

运行 PyTorch 版本：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo1/run.py pytorch
```

顺序运行全部版本：

```bash
UV_CACHE_DIR=/Users/andrewli/.cache/uv uv run python demos/demo1/run.py all
```

## 输出

- NumPy 图像: `demos/demo1/artifacts/numpy_fit_result.png`
- PyTorch 图像: `demos/demo1/artifacts/pytorch_fit_result.png`

两份脚本都会打印训练日志，并在最后输出拟合结果图。
