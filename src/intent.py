from typing import Dict
from . import config

def detect_intents(query: str) -> Dict[str, float]:
    q = (query or "").lower()
    scores = {"behavior": 0.0, "communication": 0.0, "aptitude": 0.0, "technical": 0.0, "sales": 0.0, "data_analyst": 0.0, "java": 0.0}
    for p in config.BEHAVIOUR_TRIGGER_PHRASES:
        if p in q: scores["behavior"] += 1.0
    for p in config.COMMUNICATION_TRIGGER_PHRASES:
        if p in q: scores["communication"] += 1.0
    for p in config.APTITUDE_TRIGGER_PHRASES:
        if p in q: scores["aptitude"] += 1.0
    if "sales" in q or "sales role" in q:
        scores["sales"] += 1.0
    if "data analyst" in q or ("sql" in q and "excel" in q):
        scores["data_analyst"] += 1.0
    if "java" in q:
        scores["java"] += 1.0
    # smooth + clip
    for k, v in list(scores.items()):
        v = min(max(v, 0.0), 3.0)
        scores[k] = v / 3.0
    # technical coarse
    if any(w in q for w in ["python","java","sql","selenium","automation","devops","cloud"]):
        scores["technical"] = max(scores["technical"], 0.5)
    return scores

def expansion_seeds(scores: Dict[str, float]) -> Dict[str, float]:
    """Return a dict of seed terms with weights."""
    seeds: Dict[str, float] = {}
    for key, weight in scores.items():
        if weight <= 0: 
            continue
        for term in config.RETRIEVAL_BOOST_SEEDS.get(key, []):
            seeds[term] = max(seeds.get(term, 0.0), weight)
    return seeds


