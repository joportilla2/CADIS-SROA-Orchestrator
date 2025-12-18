from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable,Optional
import numpy as np

@dataclass(frozen=True)
class ParamBounds:
    lam: Tuple[float, float] = (1.0, 1.8)
    gamma: Tuple[float, float] = (0.0, 4.0)
    omega: Tuple[float, float] = (0.0, 1.0)

# cadis_sroa_rep2/design.py
# -*- coding: utf-8 -*-
"""
Diseños experimentales (Latin Hypercube) para la fase de sensibilidad.

Esta versión es compatible con pipeline.run_sensitivity, que llama:

    latin_hypercube(n_samples, 3, seed=design_seed)

Es decir, el segundo argumento es el número de dimensiones (n_dims),
y el escalado a [lam_min, lam_max], [gamma_min, gamma_max],
[omega_min, omega_max] se hace dentro de pipeline.py.
"""

def latin_hypercube(n_samples: int, n_dims: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Genera un diseño Latin Hypercube en el hipercubo [0, 1]^n_dims.
    Devuelve matriz shape (n_samples, n_dims).
    """
    rng = np.random.default_rng(seed)

    cut = np.linspace(0.0, 1.0, n_samples + 1)

    u = rng.random((n_samples, n_dims))   # (n_samples, n_dims)
    a = cut[:-1]                          # (n_samples,)
    b = cut[1:]                           # (n_samples,)

    # FIX: expandimos a y (b-a) a (n_samples,1) para broadcasting correcto
    rdpoints = a[:, None] + (b - a)[:, None] * u  # (n_samples, n_dims)

    H = np.zeros_like(rdpoints)
    for j in range(n_dims):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, j]

    return H


def pareto_front(rows: Iterable[Dict], objectives: List[Tuple[str, str]]) -> List[Dict]:
    """Return non-dominated set (Pareto front).

    objectives: list of (metric_key, direction) where direction is 'max' or 'min'
    """
    rows = list(rows)
    if not rows:
        return []
    # normalize direction: convert to maximization
    def score(r, key, direction):
        v = r.get(key)
        if v is None:
            return None
        return v if direction == "max" else -v

    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            better_or_equal = True
            strictly_better = False
            for key, direction in objectives:
                sa = score(a, key, direction)
                sb = score(b, key, direction)
                if sa is None or sb is None:
                    better_or_equal = False
                    break
                if sb < sa:
                    better_or_equal = False
                    break
                if sb > sa:
                    strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front
