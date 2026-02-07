from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def write_learning_curve(
    *,
    out_png: Path,
    episodes: List[int],
    train_return: List[float],
    train_len: List[int],
    eval_return: Optional[List[float]] = None,
    eval_episodes: Optional[List[int]] = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.plot(episodes, train_return, label="train_return")
    if eval_return is not None and eval_episodes is not None and len(eval_return) == len(eval_episodes):
        ax.plot(eval_episodes, eval_return, label="eval_return")

    ax.grid(True)
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
