"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import pfl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # Auswahlregler: der Wert rastet auf einen erlaubten Wert ein, sonst wird er verworfen


SETTING_SPECS = {
    "mode_select": SettingSpec("mode", str, C.DEFAULT_MODE, choices=C.MODES),
    "sr_slider": SettingSpec("sr", float, C.DEFAULT_SIGMA_R, C.SIGMA_R_MIN, C.SIGMA_R_MAX),
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "second_select": SettingSpec("second", int, C.DEFAULT_SECOND, choices=C.SECOND_CHOICES),
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"mode": "mode_select", "sr": "sr_slider", "n": "n_select", "second": "second_select", "q": "q_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp_value(spec, value):
    """Wert auf die Reglergrenzen begrenzen bzw. auf einen erlaubten Auswahlwert prüfen; None heißt: verwerfen."""
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            raw = qp[spec.url_param]
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            try:
                value = clamp_value(spec, spec.caster(raw))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
