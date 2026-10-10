"""Einheiten des Partikelfilter-Kerns gegen Handrechnung und Randfälle."""

import numpy as np
import pytest

import pfl_algorithm as A
import pfl_constants as C
import pfl_scenario as S


def test_h_and_jacobian_hand_values(range_model):
    x = np.array([3.0, 4.0, 0.0, 0.0])
    assert A.h(x, range_model) == pytest.approx([5.0])
    assert A.jacobian(x, range_model)[0] == pytest.approx([0.6, 0.8, 0.0, 0.0])


def test_range_only_measurement_cannot_tell_mirror_images_apart(range_model):
    a = np.array([30.0, 40.0, 1.0, 0.0])
    b = np.array([30.0, -40.0, 1.0, 0.0])
    assert A.h(a, range_model) == pytest.approx(A.h(b, range_model))


def test_range_bearing_breaks_the_mirror_symmetry():
    m = A.Measurement("range_bearing", ((0.0, 0.0),), 1.0, 0.05)
    a = np.array([30.0, 40.0, 0.0, 0.0])
    b = np.array([30.0, -40.0, 0.0, 0.0])
    assert abs(A.h(a, m)[1] - A.h(b, m)[1]) > 1.0


def test_effective_sample_size_hand_example(weights_example):
    assert A.effective_sample_size(weights_example) == pytest.approx(1.0 / 0.375)
    assert A.effective_sample_size(np.full(10, 0.1)) == pytest.approx(10.0)       # gleiche Gewichte: alle Partikel zählen
    assert A.effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)  # ein Partikel trägt alles


def test_systematic_resampling_hand_example():
    w = np.array([0.5, 0.25, 0.25])
    # N = 3, u = 0.1: Stützpunkte 0,0333; 0,3667; 0,7. Summenfolge 0,5; 0,75; 1.0 -> Indizes 0, 0, 1
    assert A.systematic_resample(w, 0.1).tolist() == [0, 0, 1]
    # u = 0.9: Stützpunkte 0,3; 0,6333; 0,9667 -> 0, 1, 2
    assert A.systematic_resample(w, 0.9).tolist() == [0, 1, 2]


def test_systematic_resampling_keeps_counts_within_one_of_expectation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        N = int(rng.integers(5, 200))
        w = rng.random(N) + 1e-3
        w /= w.sum()
        idx = A.systematic_resample(w, float(rng.random()))
        counts = np.bincount(idx, minlength=N)
        assert counts.sum() == N and np.all(np.abs(counts - N * w) < 1.0 + 1e-9)


def test_pf_step_normalizes_weights_and_resamples_only_when_ess_is_low(range_model):
    F, Q = A.cv_model_2d(1.0, 0.05)
    Lq = np.linalg.cholesky(Q + 1e-9 * np.eye(4))
    rng = np.random.default_rng(1)
    particles = np.tile(np.array([30.0, 40.0, 0.0, 0.0]), (200, 1)) + rng.standard_normal((200, 4))
    w = np.full(200, 1 / 200)
    # Messung passt zur Wolke: wenig Gewichtsungleichheit, kein Resampling
    z_ok = A.h(np.array([30.0, 40.0, 0, 0]), range_model)
    p1, w1, ess1, res1 = A.pf_step(particles, w, z_ok, range_model, F, Lq, rng)
    assert w1.sum() == pytest.approx(1.0) and ess1 > 100 and not res1
    # Messung weit daneben: Gewicht kollabiert, es wird neu gezogen, danach gleiche Gewichte
    p2, w2, ess2, res2 = A.pf_step(particles, w, z_ok + 8.0, range_model, F, Lq, rng)
    assert res2 and w2 == pytest.approx(np.full(200, 1 / 200)) and ess2 < 100


def test_cv_model_has_no_axis_coupling():
    F, Q = A.cv_model_2d(1.0, 1.0)
    assert Q[0, 0] == pytest.approx(1 / 3) and Q[0, 2] == pytest.approx(1 / 2) and Q[2, 2] == pytest.approx(1.0)
    assert Q[0, 1] == 0 and Q[1, 2] == 0


def test_simulation_is_reproducible_and_sigma_scales_the_same_noise():
    a = S.simulate(7, "nur Entfernung", 3.0, 30, 0.05)
    b = S.simulate(7, "nur Entfernung", 3.0, 30, 0.05)
    assert np.array_equal(a.X, b.X) and all(np.array_equal(x, y) for x, y in zip(a.Z, b.Z))
    c = S.simulate(7, "nur Entfernung", 6.0, 30, 0.05)
    assert np.array_equal(a.X, c.X)
    n_a = a.Z[10][0] - A.h(a.X[10], a.meas[10])[0]
    n_c = c.Z[10][0] - A.h(c.X[10], c.meas[10])[0]
    assert n_c == pytest.approx(2 * n_a)


def test_second_beacon_appears_at_the_configured_second():
    tr = S.simulate(1, "nur Entfernung", 3.0, 30, 0.05)
    assert tr.meas[29].dim == 1 and tr.meas[30].dim == 2
    never = S.simulate(1, "nur Entfernung", 3.0, 0, 0.05)
    assert all(m.dim == 1 for m in never.meas)
    rb = S.simulate(1, "Entfernung und Peilung", 3.0, 30, 0.05)
    assert all(m.kind == "range_bearing" and m.dim == 2 for m in rb.meas)


def test_constants_are_consistent():
    assert C.DEFAULT_N in C.N_CHOICES and C.DEFAULT_SECOND in C.SECOND_CHOICES and C.DEFAULT_MODE in C.MODES
    assert 0 in C.SECOND_CHOICES and max(C.SECOND_CHOICES) < C.T_STEPS
    assert C.START[1] > 0 and C.BEACON_2 != C.BEACON_1
