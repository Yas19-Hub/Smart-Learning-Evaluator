"""
model.py — Logistic Regression classifier for Revon.
Trains on synthetic data at import time. No hardcoded predictions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

LABEL_MAP = {0: "Normal", 1: "Dyslexia Risk", 2: "ADHD Risk", 3: "Learning Difficulty"}
RISK_MAP  = {0: "Low",    1: "High",          2: "Moderate",  3: "Moderate"}
RISK_CLR  = {0: "#22c55e",1: "#ef4444",       2: "#f97316",   3: "#f59e0b"}

FEATURE_NAMES = [
    "reading_time", "reading_accuracy", "reaction_time",
    "missed_clicks", "memory_score", "task_completion", "error_rate",
]


def _make_dataset(n: int = 200, seed: int = 42):
    rng = np.random.RandomState(seed)
    X, y = [], []

    def add(cls, rt, ra, rxn, mc, ms, tc, er):
        for _ in range(n):
            X.append([
                rng.uniform(*rt),  rng.uniform(*ra),   rng.uniform(*rxn),
                rng.randint(*mc),  rng.uniform(*ms),   rng.uniform(*tc),
                rng.uniform(*er),
            ])
            y.append(cls)

    add(0, (25,55),  (78,100),(180,380),(0,3), (72,100),(35,70), (0,10))
    add(1, (75,160),(25,58), (280,580),(1,5), (50,78), (65,130),(12,38))
    add(2, (15,45), (50,78), (140,340),(5,11),(35,68), (18,55), (22,52))
    add(3, (70,150),(30,62), (380,700),(3,10),(25,58), (80,160),(28,58))
    return np.array(X, dtype=float), np.array(y, dtype=int)


_X, _y  = _make_dataset()
_scaler = StandardScaler().fit(_X)
_Xs     = _scaler.transform(_X)
_model  = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
_model.fit(_Xs, _y)

_Xtr,_Xte,_ytr,_yte = train_test_split(_Xs, _y, test_size=0.2, random_state=42)
_val = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(_Xtr,_ytr).score(_Xte,_yte)
print(f"[Revon] Model ready — validation accuracy: {_val:.2%}")


def predict(features: dict) -> dict:
    vec  = np.array([[features[f] for f in FEATURE_NAMES]], dtype=float)
    vs   = _scaler.transform(vec)
    cid  = int(_model.predict(vs)[0])
    prob = _model.predict_proba(vs)[0]
    conf = round(float(np.max(prob)) * 100, 1)

    print("\n[Revon] Feature vector:")
    for name, val in zip(FEATURE_NAMES, vec[0]):
        print(f"  {name:<22} = {val:.2f}")
    print(f"[Revon] Prediction -> {LABEL_MAP[cid]}  ({conf}%)")

    return {
        "prediction": LABEL_MAP[cid],
        "risk_level": RISK_MAP[cid],
        "risk_color": RISK_CLR[cid],
        "confidence": conf,
        "class_id":   cid,
        "all_probs":  {LABEL_MAP[i]: round(float(p)*100,1) for i,p in enumerate(prob)},
    }
