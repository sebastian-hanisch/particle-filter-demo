"""Orakel-Test particle-filter-demo: der Partikelfilter gegen ANDERE Rechenwege.

1. Linear-gaußscher Grenzfall: viele Partikel gegen den exakten Kalman-Filter,
2. UKF gleich EKF im linearen Fall (Sigma-Punkte sind dort exakt),
3. eine Messung gegen den exakten Posterior auf einem Gitter, einschließlich der zwei Gipfel (je etwa die Hälfte der Masse),
4. effektive Stichprobengröße gegen die geschlossene Formel,
5. Masse auf den beiden möglichen Orten gegen eine Schleife,
6. die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
"""

import numpy as np
import pytest

import pfl_algorithm as A
import pfl_constants as C
import pfl_evaluation as ev
import pfl_scenario as S


def _linear_track(seed, T=25):
    rng = np.random.default_rng(seed)
    F, Q = A.cv_model_2d(1.0, 0.3)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    x = np.array([0.0, 0.0, 3.0, 1.0])
    X, Z = [], []
    for _ in range(T):
        X.append(x)
        Z.append(x[:2] + 2.0 * rng.standard_normal(2))
        x = F @ x + L @ rng.standard_normal(4)
    return np.array(X), Z


def test_particle_filter_converges_to_the_kalman_filter_in_the_linear_gaussian_case():
    m = A.Measurement("position", (), 2.0)
    F, Q = A.cv_model_2d(1.0, 0.3)
    Lq = np.linalg.cholesky(Q + 1e-9 * np.eye(4))
    x0, P0 = np.array([0.0, 0.0, 3.0, 1.0]), np.diag([4.0, 4.0, 1.0, 1.0])
    for seed in range(3):
        X, Z = _linear_track(seed)
        x, P = x0.copy(), P0.copy()
        kf = [x.copy()]
        for t in range(1, len(Z)):
            x, P = A.predict(x, P, F, Q)
            x, P = A.ekf_update(x, P, Z[t], m)         # Messung ist linear: der EKF ist hier der exakte Kalman-Filter
            kf.append(x.copy())
        rng = np.random.default_rng(seed)
        N = 40000
        parts = x0 + rng.standard_normal((N, 4)) @ np.linalg.cholesky(P0).T
        w = np.full(N, 1.0 / N)
        pf = [w @ parts]
        for t in range(1, len(Z)):
            parts, w, _, _ = A.pf_step(parts, w, Z[t], m, F, Lq, rng)
            pf.append(w @ parts)
        diff = np.abs(np.array(pf)[:, :2] - np.array(kf)[:, :2])
        assert diff.max() < 0.35, (seed, diff.max())       # Messrauschen 2 m: Abweichung klein gegen die Unsicherheit


def test_ukf_equals_ekf_when_the_measurement_is_linear():
    m = A.Measurement("position", (), 2.0)
    F, Q = A.cv_model_2d(1.0, 0.3)
    X, Z = _linear_track(5)
    xe, Pe = np.array([0.0, 0.0, 3.0, 1.0]), np.diag([4.0, 4.0, 1.0, 1.0])
    xu, Pu = xe.copy(), Pe.copy()
    for t in range(1, len(Z)):
        xe, Pe = A.predict(xe, Pe, F, Q)
        xu, Pu = A.predict(xu, Pu, F, Q)
        xe, Pe = A.ekf_update(xe, Pe, Z[t], m)
        xu, Pu = A.ukf_update(xu, Pu, Z[t], m)
    assert xu == pytest.approx(xe, abs=1e-6) and Pu == pytest.approx(Pe, abs=1e-6)


def _grid_posterior(x_prior, std, z, sigma_r, n=900, span=4.5):
    """Exakter Posterior der Position nach einer Entfernungsmessung von einer Bake im Ursprung (Gauß-Prior mit Streuung (sx, sy)) auf einem Gitter."""
    gx = np.linspace(x_prior[0] - span * std[0], x_prior[0] + span * std[0], n)
    gy = np.linspace(x_prior[1] - span * std[1], x_prior[1] + span * std[1], n)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    prior = np.exp(-0.5 * (((GX - x_prior[0]) / std[0]) ** 2 + ((GY - x_prior[1]) / std[1]) ** 2))
    lik = np.exp(-0.5 * (z - np.hypot(GX, GY)) ** 2 / sigma_r ** 2)
    w = prior * lik
    return GX, GY, w / w.sum()


def test_one_range_measurement_against_the_exact_grid_posterior_including_both_modes():
    sigma_r = 3.0
    m = A.Measurement("range", ((0.0, 0.0),), sigma_r)
    x_true = np.array([20.0, 60.0, 0.0, 0.0])
    z = A.h(x_true, m)
    x_prior = np.array([20.0, 0.0])
    std = np.array([5.0, 60.0])
    GX, GY, W = _grid_posterior(x_prior, std, z[0], sigma_r)
    exact_up = float(W[GY > 0].sum())
    exact_mean = np.array([(W * GX).sum(), (W * GY).sum()])
    rng = np.random.default_rng(3)
    N = 300000
    parts = np.c_[x_prior[0] + std[0] * rng.standard_normal(N), x_prior[1] + std[1] * rng.standard_normal(N), np.zeros(N), np.zeros(N)]
    ll = -0.5 * (z[0] - A.h_batch(parts, m)[:, 0]) ** 2 / sigma_r ** 2
    w = np.exp(ll - ll.max())
    w /= w.sum()
    pf_up = float(w[parts[:, 1] > 0].sum())
    assert exact_up == pytest.approx(0.5, abs=0.02)             # beide Gipfel gleich schwer (Prior symmetrisch)
    assert pf_up == pytest.approx(exact_up, abs=0.01)
    assert (w @ parts[:, :2]) == pytest.approx(exact_mean, abs=0.5)
    # ein Gauß-Mittelwert läge dazwischen: y nahe 0, die Partikel dagegen auf beiden Seiten bei etwa +-60
    big = w > 0.2 * w.max()
    assert np.abs(parts[big, 1]).mean() > 45.0 and abs(float(w @ parts[:, 1])) < 20.0


def test_effective_sample_size_matches_the_closed_form_for_a_gaussian_prior():
    rng = np.random.default_rng(4)
    for d in (1, 2, 4):
        for ratio in (1.0, 0.5):
            x = rng.standard_normal((200000, d))
            w = np.exp(-0.5 * (x ** 2).sum(axis=1) / ratio ** 2)
            w /= w.sum()
            frac = A.effective_sample_size(w) / len(w)
            assert frac == pytest.approx(ev.ess_fraction_closed_form(d, ratio), rel=0.05), (d, ratio)


def test_candidate_mass_functions_against_a_plain_loop():
    rng = np.random.default_rng(6)
    parts = np.c_[rng.uniform(-100, 100, 500), rng.uniform(-100, 100, 500), np.zeros(500), np.zeros(500)]
    w = rng.random(500)
    w /= w.sum()
    x_true = np.array([10.0, 60.0, 0, 0])
    ref = 0.0
    for p, wi in zip(parts, w):
        if np.hypot(p[0] - 10.0, p[1] - 60.0) < C.OK_ERR or np.hypot(p[0] - 10.0, p[1] + 60.0) < C.OK_ERR:
            ref += wi
    assert ev.candidate_mass_particles(parts, w, x_true) == pytest.approx(ref)


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11):
        s = ev.Settings(seed=seed, n=500)
        a = ev.analyse(s)
        tr = S.simulate(seed, s.mode, s.sigma_r, s.second, s.q)
        assert a.track.X == pytest.approx(tr.X)
        for k, est in (("ekf", a.ekf), ("ukf", a.ukf), ("pf", a.pf.x)):
            assert a.final[k] == pytest.approx(np.hypot(*(est[-1, :2] - tr.X[-1, :2])))
            e = np.hypot(est[C.N_BURN:, 0] - tr.X[C.N_BURN:, 0], est[C.N_BURN:, 1] - tr.X[C.N_BURN:, 1])
            assert a.rmse[k] == pytest.approx(np.sqrt(np.mean(e ** 2)))
            assert a.found[k] == (a.final[k] < C.OK_ERR)
        assert np.all(a.pf.ess >= 1.0 - 1e-9) and np.all(a.pf.ess <= s.n + 1e-9)
        ref = ev.candidate_mass_particles
        assert np.all((a.pf.cand_mass >= 0) & (a.pf.cand_mass <= 1.0 + 1e-12)) and ref is not None
