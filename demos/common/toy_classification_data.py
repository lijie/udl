"""Synthetic classification datasets shared by demo2 scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetBundle:
    """Container for one train/test split and its preprocessing statistics."""

    name: str
    title: str
    train_raw: np.ndarray
    y_train: np.ndarray
    test_raw: np.ndarray
    y_test: np.ndarray
    train_x: np.ndarray
    test_x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    metadata: dict[str, float | int | str]


DATASET_SPECS = [
    {
        "name": "moons",
        "title": "Moons / 月牙形数据集",
        "generator": "moons",
        "noise": 0.12,
    },
    {
        "name": "circles",
        "title": "Circles / 环形数据集",
        "generator": "circles",
        "noise": 0.08,
        "factor": 0.45,
    },
    {
        "name": "spiral",
        "title": "Spiral / 螺旋线数据集",
        "generator": "spiral",
        "noise": 0.08,
        "rotations": 1.75,
    },
]


def make_moons(n_samples: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate a classic two-moons binary classification dataset."""
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, np.pi, n_outer)
    theta_inner = rng.uniform(0.0, np.pi, n_inner)

    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    inner = np.column_stack([1.0 - np.cos(theta_inner), 0.5 - np.sin(theta_inner)])

    x = np.vstack([outer, inner])
    y = np.concatenate([
        np.zeros(n_outer, dtype=np.float64),
        np.ones(n_inner, dtype=np.float64),
    ])

    x += rng.normal(0.0, noise, size=x.shape)
    return x, y


def make_circles(
    n_samples: int,
    noise: float,
    factor: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles for binary classification."""
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, 2.0 * np.pi, n_outer)
    theta_inner = rng.uniform(0.0, 2.0 * np.pi, n_inner)

    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    inner = factor * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])

    x = np.vstack([outer, inner])
    y = np.concatenate([
        np.zeros(n_outer, dtype=np.float64),
        np.ones(n_inner, dtype=np.float64),
    ])

    x += rng.normal(0.0, noise, size=x.shape)
    return x, y


def make_spiral(
    n_samples: int,
    noise: float,
    rotations: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving spiral arms."""
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0

    radius0 = np.linspace(0.05, 1.0, n_class0)
    radius1 = np.linspace(0.05, 1.0, n_class1)
    theta0 = rotations * 2.0 * np.pi * radius0
    theta1 = rotations * 2.0 * np.pi * radius1 + np.pi

    class0 = np.column_stack([radius0 * np.cos(theta0), radius0 * np.sin(theta0)])
    class1 = np.column_stack([radius1 * np.cos(theta1), radius1 * np.sin(theta1)])

    x = np.vstack([class0, class1])
    y = np.concatenate([
        np.zeros(n_class0, dtype=np.float64),
        np.ones(n_class1, dtype=np.float64),
    ])

    x += rng.normal(0.0, noise, size=x.shape)
    return x, y


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using train statistics only."""
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    train_scaled = (train_x - mean) / std
    test_scaled = (test_x - mean) / std
    return train_scaled, test_scaled, mean, std


def transform_points(points: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the train-set standardization parameters to arbitrary points."""
    return (points - mean) / std


def build_demo2_datasets(
    n_train: int,
    n_test: int,
    seed: int,
    selected_names: list[str] | None = None,
) -> list[DatasetBundle]:
    """Build train/test splits for the requested demo2 datasets."""
    total = n_train + n_test
    selected = set(selected_names) if selected_names else None
    datasets: list[DatasetBundle] = []

    for index, spec in enumerate(DATASET_SPECS):
        if selected is not None and spec["name"] not in selected:
            continue

        rng = np.random.default_rng(seed + 1000 * (index + 1))
        if spec["generator"] == "moons":
            x, y = make_moons(total, noise=float(spec["noise"]), rng=rng)
        elif spec["generator"] == "circles":
            x, y = make_circles(
                total,
                noise=float(spec["noise"]),
                factor=float(spec["factor"]),
                rng=rng,
            )
        else:
            x, y = make_spiral(
                total,
                noise=float(spec["noise"]),
                rotations=float(spec["rotations"]),
                rng=rng,
            )

        perm = rng.permutation(total)
        x = x[perm]
        y = y[perm]

        train_raw = x[:n_train]
        test_raw = x[n_train:]
        y_train = y[:n_train].reshape(-1, 1)
        y_test = y[n_train:].reshape(-1, 1)

        train_x, test_x, mean, std = standardize_train_test(train_raw, test_raw)
        metadata = {
            "noise": float(spec["noise"]),
            "n_train": int(n_train),
            "n_test": int(n_test),
        }
        if "factor" in spec:
            metadata["factor"] = float(spec["factor"])
        if "rotations" in spec:
            metadata["rotations"] = float(spec["rotations"])

        datasets.append(
            DatasetBundle(
                name=str(spec["name"]),
                title=str(spec["title"]),
                train_raw=train_raw.astype(np.float64),
                y_train=y_train.astype(np.float64),
                test_raw=test_raw.astype(np.float64),
                y_test=y_test.astype(np.float64),
                train_x=train_x.astype(np.float64),
                test_x=test_x.astype(np.float64),
                mean=mean.astype(np.float64),
                std=std.astype(np.float64),
                metadata=metadata,
            )
        )

    return datasets
