"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

import time
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np

import pfl_algorithm as A
import pfl_constants as C
import pfl_scenario as S


@dataclass(frozen=True)
class Settings:
    mode: str = C.DEFAULT_MODE
    sigma_r: float = C.DEFAULT_SIGMA_R
    n: int = C.DEFAULT_N
    second: int = C.DEFAULT_SECOND          # Sekunde, ab der die zweite Bake misst; 0 = nie
    q: float = C.DEFAULT_Q
    seed: int = C.DEFAULT_SEED


def prior(settings):
    x0 = np.array([C.START[0], 0.0, C.SPEED, 0.0])
    P0 = np.diag([C.X_PRIOR_STD ** 2, C.Y_PRIOR_STD ** 2, C.V_PRIOR_STD ** 2, C.V_PRIOR_STD ** 2])
    return x0, P0


def run_gaussian(track, kind, settings):
    """EKF oder UKF über eine ganze Fahrt. Gibt (x (T,4), P (T,4,4)) zurück."""
    F, Q = A.cv_model_2d(C.DT, settings.q)
    x, P = prior(settings)
    xs, Ps = np.zeros((C.T_STEPS, 4)), np.zeros((C.T_STEPS, 4, 4))
    update = A.ekf_update if kind == "ekf" else A.ukf_update
    for t in range(C.T_STEPS):
        if t > 0:
            x, P = A.predict(x, P, F, Q)
            x, P = update(x, P, track.Z[t], track.meas[t])
        xs[t], Ps[t] = x, P
    return xs, Ps


@dataclass(frozen=True)
class PFRun:
    x: np.ndarray            # (T, 4) gewichteter Mittelwert
    y_mass_pos: np.ndarray   # (T,) Gewichtsanteil der Partikel mit y > 0
    ess: np.ndarray          # (T,) effektive Stichprobengröße vor dem Resampling
    resampled: np.ndarray    # (T,) bool
    cloud: np.ndarray        # (T, S, 2) Stichprobe der Wolke (nach Gewichten gezogen) zur Darstellung
    cand_mass: np.ndarray    # (T,) Gewichtsanteil innerhalb OK_ERR um einen der beiden spiegelbildlichen Orte (x, +y) und (x, -y)
    seconds: float


def run_particle_filter(track, settings, rng_seed=0, keep_cloud=True):
    F, Q = A.cv_model_2d(C.DT, settings.q)
    Lq = np.linalg.cholesky(Q + 1e-9 * np.eye(4))
    x0, P0 = prior(settings)
    rng = np.random.default_rng(rng_seed + settings.seed)
    N = settings.n
    particles = x0 + rng.standard_normal((N, 4)) @ np.linalg.cholesky(P0).T
    w = np.full(N, 1.0 / N)
    T = C.T_STEPS
    xs, ym, cm = np.zeros((T, 4)), np.zeros(T), np.zeros(T)
    ess, res = np.zeros(T), np.zeros(T, bool)
    S_ = min(N, C.PARTICLE_SHOW)
    cloud = np.zeros((T, S_, 2)) if keep_cloud else None
    disp_rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    for t in range(T):
        if t > 0:
            particles, w, ess[t], res[t] = A.pf_step(particles, w, track.Z[t], track.meas[t], F, Lq, rng)
        else:
            ess[t] = A.effective_sample_size(w)
        xs[t] = w @ particles
        ym[t] = float(w[particles[:, 1] > 0].sum())
        cm[t] = candidate_mass_particles(particles, w, track.X[t])
        if keep_cloud:
            idx = disp_rng.choice(N, S_, p=w / w.sum())
            cloud[t] = particles[idx, :2]
    return PFRun(xs, ym, ess, res, cloud, cm, time.perf_counter() - t0)


def candidate_mass_particles(particles, w, x_true):
    """Gewichtsanteil der Partikel innerhalb OK_ERR um einen der beiden spiegelbildlichen Orte (x, +y) und (x, -y) der wahren Position."""
    near_a = np.hypot(particles[:, 0] - x_true[0], particles[:, 1] - x_true[1]) < C.OK_ERR
    near_b = np.hypot(particles[:, 0] - x_true[0], particles[:, 1] + x_true[1]) < C.OK_ERR
    return float(w[near_a | near_b].sum())


def candidate_mass_gaussian(mean, cov, x_true, rng, n=4000):
    """Dasselbe für eine Gauß-Verteilung (Stichprobe aus N(mean[:2], cov[:2, :2]), feste Zufallszahlen über `rng`)."""
    smp = rng.multivariate_normal(mean[:2], cov[:2, :2] + 1e-9 * np.eye(2), n)
    near_a = np.hypot(smp[:, 0] - x_true[0], smp[:, 1] - x_true[1]) < C.OK_ERR
    near_b = np.hypot(smp[:, 0] - x_true[0], smp[:, 1] + x_true[1]) < C.OK_ERR
    return float(np.mean(near_a | near_b))


def _gauss_side(mean_y, var_y):
    return NormalDist(mean_y, max(var_y, 1e-12) ** 0.5).cdf(0.0)      # P(y < 0)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    track: S.Track
    ekf: np.ndarray
    ukf: np.ndarray
    ukf_P: np.ndarray
    ekf_P: np.ndarray
    pf: PFRun
    err: dict               # Methode -> (T,) Positionsfehler
    rmse: dict
    final: dict             # Methode -> Endfehler
    found: dict             # Methode -> bool (Endfehler unter OK_ERR)
    side_true: np.ndarray   # (T,) PF: Masse auf der wahren Straßenseite (y > 0, die Wahrheit liegt bei y = 60)
    side_ukf: np.ndarray    # (T,) UKF: Gauß-Wahrscheinlichkeit für y > 0
    cand_ukf: np.ndarray    # (T,) UKF: Wahrscheinlichkeit auf den beiden möglichen Orten


def analyse(settings, keep_cloud=True):
    tr = S.simulate(settings.seed, settings.mode, settings.sigma_r, settings.second, settings.q)
    ekf, ekf_P = run_gaussian(tr, "ekf", settings)
    ukf, ukf_P = run_gaussian(tr, "ukf", settings)
    pf = run_particle_filter(tr, settings, keep_cloud=keep_cloud)
    pos = tr.X[:, :2]
    ests = {"ekf": ekf, "ukf": ukf, "pf": pf.x}
    err = {k: np.linalg.norm(e[:, :2] - pos, axis=1) for k, e in ests.items()}
    sl = slice(C.N_BURN, None)
    rmse = {k: float(np.sqrt(np.mean(e[sl] ** 2))) for k, e in err.items()}
    final = {k: float(e[-1]) for k, e in err.items()}
    found = {k: final[k] < C.OK_ERR for k in err}
    side_ukf = np.array([1.0 - _gauss_side(ukf[t, 1], ukf_P[t, 1, 1]) for t in range(C.T_STEPS)])
    cg = np.random.default_rng(5)
    cand_ukf = np.array([candidate_mass_gaussian(ukf[t], ukf_P[t], tr.X[t], cg) for t in range(C.T_STEPS)])
    return Analysis(settings, tr, ekf, ukf, ukf_P, ekf_P, pf, err, rmse, final, found, pf.y_mass_pos, side_ukf, cand_ukf)


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def sweep_particles(settings):
    """Partikelzahl variieren bei eindeutiger Messung (Entfernung und Peilung): Positionsfehler und Rechenzeit, EKF und UKF als Referenz."""
    base = replace(settings, mode="Entfernung und Peilung", second=0)
    rows = []
    gauss = [analyse(replace(base, seed=s, n=100), keep_cloud=False) for s in C.SWEEP_SEEDS]
    ref = {"ekf": float(np.median([a.rmse["ekf"] for a in gauss])), "ukf": float(np.median([a.rmse["ukf"] for a in gauss]))}
    for n in C.N_CHOICES:
        res = [analyse(replace(base, seed=s, n=n), keep_cloud=False) for s in C.SWEEP_SEEDS]
        rows.append({"n": n, "rmse": float(np.median([a.rmse["pf"] for a in res])), "seconds": float(np.median([a.pf.seconds for a in res]))})
    return {"rows": rows, **ref}


def sweep_ambiguity(settings):
    """Mehrdeutige Entfernungsmessung: wie oft findet jedes Verfahren die wahre Seite, je nachdem wann die zweite Bake dazukommt (0 = nie)."""
    base = replace(settings, mode="nur Entfernung")
    rows = []
    for sec in C.SECOND_CHOICES:
        res = [analyse(replace(base, seed=s, second=sec), keep_cloud=False) for s in C.SWEEP_SEEDS]
        t_before = (sec - 1) if sec > 0 else C.T_STEPS - 1
        rows.append({
            "second": sec,
            "found": {k: float(np.mean([a.found[k] for a in res])) for k in ("ekf", "ukf", "pf")},
            "median": {k: float(np.median([a.final[k] for a in res])) for k in ("ekf", "ukf", "pf")},
            "pf_side": float(np.mean([a.side_true[t_before] for a in res])),
            "ukf_side": float(np.mean([a.side_ukf[t_before] for a in res])),
            "pf_cand": float(np.mean([a.pf.cand_mass[t_before] for a in res])),
            "ukf_cand": float(np.mean([a.cand_ukf[t_before] for a in res])),
        })
    return rows


def ess_fraction_closed_form(d, ratio):
    """Erwartete ESS/N beim ersten Gewichten für Gauß-Prior N(0, I_d) und Gauß-Messung mit Streuung ratio je Komponente, Messwert = Prior-Mittel:
    ESS/N = ((1 + 2 s^-2) / (1 + s^-2)^2)^(d/2)  (für großes N)."""
    r = 1.0 / ratio ** 2
    return float(((1.0 + 2.0 * r) / (1.0 + r) ** 2) ** (d / 2.0))


def sweep_dimension(settings):
    """Partikelverarmung mit der Dimension: Anteil der effektiven Partikel nach EINEM Gewichtungsschritt, Monte Carlo gegen geschlossene Formel."""
    rng = np.random.default_rng(777)
    rows = []
    for ratio in C.DIM_RATIOS:
        for d in C.DIM_SWEEP:
            vals = []
            for _ in range(C.DIM_REPEATS):
                x = rng.standard_normal((C.DIM_PARTICLES, d))
                w = np.exp(-0.5 * (x ** 2).sum(axis=1) / ratio ** 2)
                w /= w.sum()
                vals.append(A.effective_sample_size(w) / C.DIM_PARTICLES)
            rows.append({"ratio": ratio, "d": d, "mc": float(np.mean(vals)), "formula": ess_fraction_closed_form(d, ratio)})
    return rows
