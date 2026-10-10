"""Szenario: ein Fahrzeug fährt eine gerade Straße entlang; eine Bake (optional eine zweite) misst Entfernung (und Peilung)."""

from dataclasses import dataclass

import numpy as np

import pfl_algorithm as A
import pfl_constants as C


@dataclass(frozen=True)
class Track:
    t: np.ndarray
    X: np.ndarray         # (T, 4) wahrer Zustand
    Z: list               # Länge T: Messvektor je Schritt (Zeile 0 unbenutzt)
    meas: list            # Länge T: das zugehörige A.Measurement je Schritt


def measurement_at(mode, t, second_from, sigma_r):
    """Messmodell zum Zeitpunkt t: Entfernung+Peilung (eine Bake) oder Entfernungen zu den aktiven Baken."""
    if mode == "Entfernung und Peilung":
        return A.Measurement("range_bearing", (C.BEACON_1,), sigma_r, np.deg2rad(C.SIGMA_B_DEG))
    beacons = (C.BEACON_1,) + ((C.BEACON_2,) if second_from > 0 and t >= second_from else ())
    return A.Measurement("range", beacons, sigma_r)


def simulate(seed, mode, sigma_r, second_from, q):
    """Wahre Fahrt und Messungen. Alle Zufallszahlen kommen in fester Reihenfolge aus einem Generator."""
    rng = np.random.default_rng(seed)
    n_proc = rng.standard_normal((C.T_STEPS, 4))
    n_meas = rng.standard_normal((C.T_STEPS, 2))
    F, Q = A.cv_model_2d(C.DT, q)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    X = np.zeros((C.T_STEPS, 4))
    x = np.array([C.START[0], C.START[1], C.SPEED, 0.0])
    for k in range(C.T_STEPS):
        X[k] = x
        x = F @ x + L @ n_proc[k]
    Z, M = [], []
    for k in range(C.T_STEPS):
        m = measurement_at(mode, k, second_from, sigma_r)
        z = A.h(X[k], m)
        noise = np.array([sigma_r * n_meas[k, 0], m.sigma_b * n_meas[k, 1]]) if m.kind == "range_bearing" else sigma_r * np.resize(n_meas[k], m.dim)
        if m.kind == "range" and m.dim == 2:
            noise = sigma_r * n_meas[k]
        Z.append(z + noise)
        M.append(m)
    return Track(np.arange(C.T_STEPS) * C.DT, X, Z, M)
