from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from .argumentation import AF, grounded_extension, induced_defeat_graph
from .reputation import BetaRep, katz_centrality, fuse_reputation

@dataclass
class SROAConfig:
    mode: str = "pure"  # pure | hybrid
    lam: float = 1.25
    gamma: float = 2.0
    omega: float = 0.8
    forgetting: float = 0.01
    eps: float = 1e-6

class SROA:
    def __init__(self, agent_ids: List[str], config: Optional[SROAConfig] = None):
        self.cfg = config or SROAConfig()
        self.hist: Dict[str, BetaRep] = {aid: BetaRep() for aid in agent_ids}

    def decide(self, af: AF, agent_edges: List[Tuple[int, int]], agent_index: Dict[str, int]) -> Tuple[str, Dict[str, Any]]:
        ctx_vec = katz_centrality(num_nodes=len(agent_index), edges=agent_edges, alpha=0.1, beta=1.0)
        ctx = {aid: float(ctx_vec[idx]) for aid, idx in agent_index.items()}
        rep = fuse_reputation(self.hist, ctx, omega=self.cfg.omega)

        weights: Dict[str, float] = {}
        for a_id, a in af.args.items():
            r = rep.get(a.agent_id, 0.5)
            weights[a_id] = (r + self.cfg.eps) ** self.cfg.gamma * (a.confidence + self.cfg.eps)

        if self.cfg.mode == "hybrid":
            af_used = induced_defeat_graph(af, weights, self.cfg.lam)
        else:
            af_used = af

        G = grounded_extension(af_used)

        score: Dict[str, float] = {}
        for a_id in G:
            a = af.args[a_id]
            score.setdefault(a.hypothesis, 0.0)
            score[a.hypothesis] += weights[a_id] if self.cfg.mode == "hybrid" else a.confidence

        if not score:
            best = max(af.args.values(), key=lambda x: x.confidence)
            pred = best.hypothesis
        else:
            pred = max(score.items(), key=lambda kv: kv[1])[0]

        expl = {
            "accepted_args_count": len(G),
            "candidate_hypotheses_count": len({a.hypothesis for a in af.args.values()}),
            "support_score": {k: float(v) for k, v in score.items()},
            "mode": self.cfg.mode,
            "lambda": self.cfg.lam,
            "gamma": self.cfg.gamma,
            "omega": self.cfg.omega,
        }
        if self.cfg.mode == "hybrid":
            expl["attacks_total"] = len(af.attacks)
            expl["defeats_total"] = len(af_used.attacks)
            expl["filtered_attacks"] = len(af.attacks) - len(af_used.attacks)
        return pred, expl

    def update_reputation(self, agent_pred: Dict[str, str], ground_truth: str) -> None:
        for aid, pred in agent_pred.items():
            self.hist[aid].update(success=(pred == ground_truth), forgetting=self.cfg.forgetting)
