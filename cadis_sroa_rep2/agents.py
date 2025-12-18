from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

from dataclasses import dataclass
from typing import List, Tuple
import random
import uuid

@dataclass
class Proposal:
    arg_id: str
    hypothesis: str
    confidence: float

class SpecialistAgent:
    def __init__(self, agent_id: str, accuracy: float,
                 conf_correct: Tuple[float, float], conf_wrong: Tuple[float, float],
                 attack_rate: float, target_expert_bias: float = 0.0, overconfident: bool = False):
        self.agent_id = agent_id
        self.accuracy = accuracy
        self.conf_correct = conf_correct
        self.conf_wrong = conf_wrong
        self.attack_rate = attack_rate
        self.target_expert_bias = target_expert_bias
        self.overconfident = overconfident

    def propose(self, hypotheses: List[str], ground_truth: str) -> Proposal:
        correct = (random.random() < self.accuracy)
        if correct:
            h = ground_truth
            lo, hi = self.conf_correct
        else:
            h = random.choice([x for x in hypotheses if x != ground_truth])
            lo, hi = self.conf_wrong
        conf = random.uniform(lo, hi)
        if self.overconfident and not correct:
            conf = max(conf, 0.80)
        return Proposal(arg_id=f"ARG-{self.agent_id}-{uuid.uuid4().hex[:8]}", hypothesis=h, confidence=conf)
