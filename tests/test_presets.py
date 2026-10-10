"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import pfl_constants as C
import pfl_evaluation as ev
import pfl_presets as P


def _settings_from(p):
    return ev.Settings(mode=p["mode"], sigma_r=p["sr"], n=p["n"], second=p["second"], q=p["q"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert p["mode"] in C.MODES and C.SIGMA_R_MIN <= p["sr"] <= C.SIGMA_R_MAX, name
        assert p["n"] in C.N_CHOICES and p["second"] in C.SECOND_CHOICES, name
        assert C.Q_MIN <= p["q"] <= C.Q_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standard (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers_and_snaps_choices():
    q = P.SETTING_SPECS["q_slider"]
    assert P.clamp_value(q, 9.0) == C.Q_MAX and P.clamp_value(q, 0.0) == C.Q_MIN
    n = P.SETTING_SPECS["n_select"]
    assert P.clamp_value(n, 1000) == 1000 and P.clamp_value(n, 999) is None
    mode = P.SETTING_SPECS["mode_select"]
    assert P.clamp_value(mode, "nur Entfernung") == "nur Entfernung" and P.clamp_value(mode, "nix") is None


def test_the_presets_show_what_their_help_text_claims():
    assert C.PRESETS["Mehrdeutigkeit bleibt"]["second"] == 0 and C.PRESETS["Mehrdeutigkeit bleibt"]["mode"] == "nur Entfernung"
    assert C.PRESETS["Eindeutige Messung"]["mode"] == "Entfernung und Peilung"
    assert C.PRESETS["Zu wenige Partikel"]["n"] == min(C.N_CHOICES)
