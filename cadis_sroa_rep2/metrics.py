from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from typing import Dict, List, Tuple, Any
import math

def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))

def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        if tp == 0 and (fp + fn) == 0:
            f1 = 1.0
        elif tp == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - half, center + half

def mcnemar_exact(pairs: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    b = c = 0
    for a_ok, b_ok in pairs:
        if a_ok and not b_ok:
            b += 1
        if (not a_ok) and b_ok:
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(b, c)
    from math import comb
    p = sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    p = min(1.0, 2 * p)
    return {"b": b, "c": c, "p": p}
