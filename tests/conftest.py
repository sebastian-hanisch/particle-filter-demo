"""Gemeinsame Fixtures: Mini-Beispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pfl_algorithm as A  # noqa: E402


@pytest.fixture
def range_model():
    """Eine Bake im Ursprung misst nur die Entfernung, Rauschen 1 m.

    Handrechnung am Punkt (3, 4): Entfernung 5, Jacobi-Zeile (0,6; 0,8; 0; 0).
    """
    return A.Measurement("range", ((0.0, 0.0),), 1.0)


@pytest.fixture
def weights_example():
    """Normierte Gewichte (0,5; 0,25; 0,25): ESS = 1 / (0,25 + 0,0625 + 0,0625) = 2,666..."""
    return np.array([0.5, 0.25, 0.25])
