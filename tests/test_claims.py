"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (30 feste Fahrten).

Plattformrobust: Bänder statt exakter Werte, Rechenzeiten nur der Ordnung nach (CI installiert neueste numpy/scipy)."""

from dataclasses import replace

import numpy as np
import pytest

import pfl_constants as C
import pfl_evaluation as ev


@pytest.fixture(scope="module")
def particle_sweep():
    return ev.sweep_particles(ev.Settings())


@pytest.fixture(scope="module")
def ambiguity_rows():
    return {r["second"]: r for r in ev.sweep_ambiguity(ev.Settings())}


@pytest.fixture(scope="module")
def dimension_rows():
    return {(r["ratio"], r["d"]): r for r in ev.sweep_dimension(None)}


def test_few_particles_are_worse_and_a_thousand_reach_the_gaussian_filters(particle_sweep):
    rows = {r["n"]: r for r in particle_sweep["rows"]}
    assert rows[100]["rmse"] == pytest.approx(6.1, abs=1.2) and rows[300]["rmse"] == pytest.approx(3.4, abs=0.8)
    assert rows[1000]["rmse"] == pytest.approx(2.4, abs=0.4) and rows[10000]["rmse"] == pytest.approx(2.3, abs=0.3)
    assert particle_sweep["ekf"] == pytest.approx(2.4, abs=0.4) and particle_sweep["ukf"] == pytest.approx(2.2, abs=0.4)
    assert rows[100]["rmse"] > 2 * rows[1000]["rmse"]
    assert rows[10000]["seconds"] > 3 * rows[1000]["seconds"]


def test_ekf_fails_on_the_ambiguous_measurement_ukf_and_pf_find_the_side_after_the_second_beacon(ambiguity_rows):
    r30 = ambiguity_rows[30]
    assert r30["found"]["ekf"] == 0.0
    assert r30["found"]["ukf"] == pytest.approx(0.97, abs=0.07)
    assert r30["found"]["pf"] == pytest.approx(0.77, abs=0.15)
    assert r30["found"]["ukf"] > r30["found"]["pf"] + 0.1          # bei 1000 Partikeln ist der UKF hier besser
    never = ambiguity_rows[0]
    assert never["found"]["ukf"] == 0.0 and never["found"]["pf"] < 0.4


def test_particle_cloud_keeps_mass_on_both_possible_places_the_gaussian_filter_does_not(ambiguity_rows):
    r30 = ambiguity_rows[30]
    assert r30["pf_cand"] == pytest.approx(0.82, abs=0.12)
    assert r30["ukf_cand"] == pytest.approx(0.09, abs=0.07)
    for r in ambiguity_rows.values():
        assert r["pf_cand"] > 2.0 * r["ukf_cand"]
        assert r["pf_side"] == pytest.approx(0.45, abs=0.1)         # beide Seiten bleiben offen


def test_more_particles_find_the_side_more_reliably():
    base = ev.Settings(mode="nur Entfernung", second=30)
    found = {}
    for n in (300, 1000, 3000):
        res = [ev.analyse(replace(base, n=n, seed=s), keep_cloud=False) for s in C.SWEEP_SEEDS]
        found[n] = float(np.mean([a.found["pf"] for a in res]))
    assert found[300] == pytest.approx(0.47, abs=0.15)
    assert found[1000] == pytest.approx(0.77, abs=0.15)
    assert found[3000] == pytest.approx(0.97, abs=0.1)
    assert found[300] < found[1000] < found[3000] + 0.05


def test_effective_particles_collapse_with_dimension(dimension_rows):
    assert dimension_rows[(1.0, 1)]["mc"] == pytest.approx(0.866, abs=0.02)
    assert dimension_rows[(1.0, 8)]["mc"] == pytest.approx(0.32, abs=0.03)
    assert dimension_rows[(0.3, 4)]["mc"] == pytest.approx(0.026, abs=0.006)
    assert dimension_rows[(0.3, 32)]["mc"] * C.DIM_PARTICLES < 2.0           # praktisch ein Partikel
    for ratio in C.DIM_RATIOS:
        mc = [dimension_rows[(ratio, d)]["mc"] for d in C.DIM_SWEEP]
        assert all(b <= a + 1e-9 for a, b in zip(mc, mc[1:]))


def test_sweeps_use_the_fixed_instances():
    assert len(C.SWEEP_SEEDS) == 30 and C.SWEEP_SEEDS[0] == 100000
