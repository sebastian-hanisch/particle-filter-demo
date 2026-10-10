"""Erweiterter, Unscented und Partikelfilter für Entfernungs-/Peilmessungen, numpy-only. Jede Funktion ist eine Einheit.

Zustand x = (x, y, vx, vy) in der Ebene, konstante Geschwindigkeit mit zufälliger Beschleunigung (q in m/s^2).
Das Messmodell ist eine Liste von Bake-Positionen und eine Art: 'range_bearing' (Entfernung und Peilung zur ersten Bake) oder 'range'
(Entfernungen zu allen aktiven Baken).
"""

from dataclasses import dataclass

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cv_model_2d(dt, q):
    """F und Q des Konstante-Geschwindigkeit-Modells in der Ebene, Reihenfolge (x, y, vx, vy)."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Q = np.zeros((4, 4))
    for a in (0, 1):
        Q[a, a] = q ** 2 * dt ** 3 / 3.0
        Q[a, a + 2] = Q[a + 2, a] = q ** 2 * dt ** 2 / 2.0
        Q[a + 2, a + 2] = q ** 2 * dt
    return F, Q


def predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q


@dataclass(frozen=True)
class Measurement:
    """Messmodell: 'range_bearing' (eine Bake), 'range' (Entfernungen zu `beacons`) oder 'position' (lineare Gegenprobe: die Position selbst)."""
    kind: str
    beacons: tuple
    sigma_r: float
    sigma_b: float = 0.0       # Radiant, nur range_bearing

    @property
    def dim(self):
        if self.kind in ("range_bearing", "position"):
            return 2
        return len(self.beacons)

    def R(self):
        if self.kind == "range_bearing":
            return np.diag([self.sigma_r ** 2, self.sigma_b ** 2])
        return np.eye(self.dim) * self.sigma_r ** 2


def h(x, m):
    """Messfunktion für einen Zustand x (4,)."""
    if m.kind == "position":
        return np.array(x[:2], float)
    if m.kind == "range_bearing":
        d = x[:2] - np.array(m.beacons[0])
        return np.array([np.hypot(d[0], d[1]), np.arctan2(d[1], d[0])])
    return np.array([np.hypot(x[0] - b[0], x[1] - b[1]) for b in m.beacons])


def h_batch(X, m):
    """Messfunktion für viele Zustände auf einmal, X: (N, 4) -> (N, dim)."""
    if m.kind == "position":
        return np.array(X[:, :2], float)
    if m.kind == "range_bearing":
        d = X[:, :2] - np.array(m.beacons[0])
        return np.c_[np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])]
    return np.stack([np.hypot(X[:, 0] - b[0], X[:, 1] - b[1]) for b in m.beacons], axis=1)


def innovation(z, zhat, m):
    nu = np.asarray(z, float) - zhat
    if m.kind == "range_bearing":
        nu = nu.copy()
        nu[..., 1] = wrap(nu[..., 1])
    return nu


def jacobian(x, m):
    """Jacobi-Matrix der Messfunktion (dim x 4)."""
    rows = []
    if m.kind == "position":
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    if m.kind == "range_bearing":
        d = x[:2] - np.array(m.beacons[0])
        r2 = d[0] ** 2 + d[1] ** 2
        r = np.sqrt(r2)
        rows.append(np.r_[d[0] / r, d[1] / r, 0, 0])
        rows.append(np.r_[-d[1] / r2, d[0] / r2, 0, 0])
    else:
        for b in m.beacons:
            d = x[:2] - np.array(b)
            rows.append(np.r_[d / np.hypot(d[0], d[1]), 0, 0])
    return np.array(rows)


def ekf_update(x, P, z, m):
    """EKF-Korrektur mit Linearisierung an der Vorhersage (Joseph-Form)."""
    R = m.R()
    H = jacobian(x, m)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(4) - K @ H
    return x + K @ innovation(z, h(x, m), m), I_KH @ P @ I_KH.T + K @ R @ K.T


def sigma_points(x, P, alpha=1.0, beta=2.0, kappa=1.0):
    n = len(x)
    lam = alpha ** 2 * (n + kappa) - n
    L = np.linalg.cholesky((n + lam) * P + 1e-9 * np.eye(n))
    pts = np.vstack([x, x + L.T, x - L.T])
    wm = np.full(2 * n + 1, 0.5 / (n + lam))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + 1 - alpha ** 2 + beta
    return pts, wm, wc


def ukf_update(x, P, z, m):
    """UKF-Korrektur mit Sigma-Punkten statt Linearisierung."""
    pts, wm, wc = sigma_points(x, P)
    Z = h_batch(pts, m)
    if m.kind == "range_bearing":
        Z[:, 1] = Z[0, 1] + wrap(Z[:, 1] - Z[0, 1])
    zm = wm @ Z
    dz = Z - zm
    S = (wc[:, None] * dz).T @ dz + m.R()
    Pxz = ((pts - x).T * wc) @ dz
    K = Pxz @ np.linalg.inv(S)
    return x + K @ innovation(z, zm, m), P - K @ S @ K.T


def systematic_resample(w, u):
    """Systematisches Resampling: ein Zufallswert `u` in [0, 1), N gleichmäßig verteilte Stützpunkte auf der kumulierten Gewichtsfolge. Gibt die Indizes zurück."""
    N = len(w)
    positions = (u + np.arange(N)) / N
    cum = np.cumsum(w)
    cum[-1] = 1.0
    return np.minimum(np.searchsorted(cum, positions), N - 1)


def effective_sample_size(w):
    """1 / sum(w^2) bei normierten Gewichten: wie viele gleich gewichtete Partikel die gewichtete Wolke wert ist."""
    return float(1.0 / np.sum(np.asarray(w) ** 2))


def pf_step(particles, w, z, m, F, Q_chol, rng, resample_below=0.5):
    """Ein Schritt des Bootstrap-Partikelfilters: vorhersagen (Rauschen ziehen), gewichten (Messwahrscheinlichkeit), bei kleiner ESS neu ziehen.

    Gibt (Partikel, Gewichte, ESS vor dem Resampling, Resampling ja/nein) zurück.
    """
    N = len(particles)
    particles = particles @ F.T + rng.standard_normal((N, 4)) @ Q_chol.T
    nu = innovation(z, h_batch(particles, m), m)
    R_inv = np.linalg.inv(m.R())
    ll = -0.5 * np.einsum("ni,ij,nj->n", nu, R_inv, nu)
    w = w * np.exp(ll - ll.max())
    s = w.sum()
    w = w / s if s > 0 else np.full(N, 1.0 / N)
    ess = effective_sample_size(w)
    resampled = ess < resample_below * N
    if resampled:
        idx = systematic_resample(w, rng.random())
        particles, w = particles[idx], np.full(N, 1.0 / N)
    return particles, w, ess, resampled
