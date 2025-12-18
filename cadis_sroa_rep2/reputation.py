from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

@dataclass
class BetaRep:
    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def update(self, success: bool, forgetting: float = 0.0) -> None:
        f = float(forgetting)
        f = 0.0 if f < 0.0 else (0.99 if f > 0.99 else f)
        self.alpha = (1.0 - f) * self.alpha
        self.beta = (1.0 - f) * self.beta
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

def katz_centrality(num_nodes: int, edges: List[Tuple[int, int]], alpha: float = 0.1, beta: float = 1.0, iters: int = 50) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=float)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            A[v, u] += 1.0  # transpose convention
    x = np.ones((num_nodes,), dtype=float)
    for _ in range(iters):
        x = alpha * (A @ x) + beta * np.ones_like(x)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def fuse_reputation(hist: Dict[str, BetaRep], ctx: Dict[str, float], omega: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for aid, b in hist.items():
        rh = b.mean()
        rc = float(ctx.get(aid, 0.0))
        r = float(omega * rh + (1.0 - omega) * rc)
        out[aid] = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
    return out
