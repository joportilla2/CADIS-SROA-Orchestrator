from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple

@dataclass(frozen=True)
class Argument:
    arg_id: str
    agent_id: str
    hypothesis: str
    confidence: float

class AF:
    def __init__(self):
        self.args: Dict[str, Argument] = {}
        self.attacks: Set[Tuple[str, str]] = set()  # (from,to)

    def add_argument(self, a: Argument) -> None:
        self.args[a.arg_id] = a

    def add_attack(self, frm: str, to: str) -> None:
        if frm in self.args and to in self.args and frm != to:
            self.attacks.add((frm, to))

    def attackers_of(self, a_id: str) -> Set[str]:
        return {u for (u, v) in self.attacks if v == a_id}

def defends(af: AF, S: Set[str], a_id: str) -> bool:
    for b in af.attackers_of(a_id):
        if not any((c, b) in af.attacks for c in S):
            return False
    return True

def grounded_extension(af: AF) -> Set[str]:
    S: Set[str] = set()
    changed = True
    while changed:
        S_new = {a for a in af.args.keys() if defends(af, S, a)}
        changed = (S_new != S)
        S = S_new
    return S

def induced_defeat_graph(af: AF, weights: Dict[str, float], lam: float) -> AF:
    rep = AF()
    for a in af.args.values():
        rep.add_argument(a)
    for (u, v) in af.attacks:
        if weights[u] >= lam * weights[v]:
            rep.add_attack(u, v)
    return rep
