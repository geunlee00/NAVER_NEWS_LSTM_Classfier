"""시각화 처리 및 평가지표 추가 모듈"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Windows 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def plot_loss(epoch_losses: List[float]) -> None:
    """epoch별 학습 loss 곡선을 그린다."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", linewidth=2)
    ax.set_title("학습 Loss 곡선")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_accuracy(report: Dict) -> None:
    """카테고리별 F1-Score 막대그래프를 그린다."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    categories = [k for k in report if k not in skip]
    f1_scores = [report[k]["f1-score"] for k in categories]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(categories, f1_scores, color="steelblue")
    ax.set_title("카테고리별 F1-Score")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def plot_report(report: Dict) -> None:
    """precision / recall / f1-score 히트맵을 그린다."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    categories = [k for k in report if k not in skip]
    metrics = ["precision", "recall", "f1-score"]

    data = np.array([[report[cat][m] for m in metrics] for cat in categories])

    fig, ax = plt.subplots(figsize=(7, len(categories) * 0.7 + 1))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title("평가지표 히트맵")

    for i in range(len(categories)):
        for j in range(len(metrics)):
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                color="white" if data[i, j] > 0.6 else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()