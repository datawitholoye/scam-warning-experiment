"""Deterministic, customer-level randomisation.

Hashing (salt + customer_id) is how production experiment platforms assign
arms: the same customer always lands in the same arm, on any server, with no
lookup table, and a new salt gives a fresh, independent split for the next test.
"""
from __future__ import annotations

import hashlib

import numpy as np


def bucket(customer_id: str, salt: str) -> float:
    """Map a customer to a stable number in [0, 1)."""
    h = hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()
    return int(h[:15], 16) / 16**15


def assign(customer_ids, salt: str, treatment_share: float = 0.5) -> np.ndarray:
    """Return 'treatment' / 'control' for each customer."""
    b = np.array([bucket(c, salt) for c in customer_ids])
    return np.where(b < treatment_share, "treatment", "control")
