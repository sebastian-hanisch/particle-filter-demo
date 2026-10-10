"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import pfl_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_cloud_map(a, t):
    """Draufsicht zum Zeitpunkt t: Partikelwolke, wahre Position und ihr Spiegelbild an der Fahrtachse, Mittelwerte der drei Filter, Baken."""
    tr = a.track
    X = tr.X
    fig = go.Figure()
    cloud = a.pf.cloud[t]
    fig.add_trace(go.Scatter(x=cloud[:, 0], y=cloud[:, 1], mode="markers", marker=dict(color=C.COLOR_PARTICLE, size=5), name="Partikel",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=X[:t + 1, 0], y=X[:t + 1, 1], mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="wahre Bahn"))
    fig.add_trace(go.Scatter(x=[X[t, 0]], y=[X[t, 1]], mode="markers", marker=dict(color=C.COLOR_TRUTH, size=14, symbol="diamond",
                                                                           line=dict(color="white", width=2)), name="wahre Position"))
    fig.add_trace(go.Scatter(x=[X[t, 0]], y=[-X[t, 1]], mode="markers", marker=dict(color="rgba(0,0,0,0)", size=14, symbol="diamond",
                                                                            line=dict(color=C.COLOR_TRUTH, width=2)), name="spiegelbildlicher Ort"))
    for name, est, color, sym in (("EKF", a.ekf, C.COLOR_EKF, "square"), ("UKF", a.ukf, C.COLOR_UKF, "triangle-up"), ("Partikelfilter (Mittel)", a.pf.x, C.COLOR_PF, "circle")):
        fig.add_trace(go.Scatter(x=[est[t, 0]], y=[est[t, 1]], mode="markers", marker=dict(color=color, size=11, symbol=sym,
                                                                                    line=dict(color="white", width=1)), name=name))
    beacons = [C.BEACON_1] + ([C.BEACON_2] if a.settings.second > 0 and t >= a.settings.second and a.settings.mode == "nur Entfernung" else [])
    fig.add_trace(go.Scatter(x=[b[0] for b in beacons], y=[b[1] for b in beacons], mode="markers", marker=dict(color=C.COLOR_BEACON, size=13, symbol="star"),
                             name="Bake(n)"))
    xmin, xmax = float(X[:, 0].min()) - 20, float(X[:, 0].max()) + 20
    ymax = float(np.abs(X[:, 1]).max()) + 40
    fig.add_trace(go.Scatter(x=[xmin, xmax], y=[-ymax, ymax], mode="markers", marker=dict(opacity=0, size=1), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title_text="x (m)")
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1)
    return _base(fig, 420)


def build_mass_chart(a):
    """Wahrscheinlichkeit auf den beiden möglichen Orten (innerhalb 15 m) über die Zeit: Partikelfilter gegen UKF."""
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.pf.cand_mass, mode="lines", line=dict(color=C.COLOR_PF, width=3), name="Partikelfilter"))
    fig.add_trace(go.Scatter(x=t, y=a.cand_ukf, mode="lines", line=dict(color=C.COLOR_UKF, width=2, dash="dash"), name="UKF"))
    s = a.settings
    if s.mode == "nur Entfernung" and s.second > 0:
        fig.add_vline(x=s.second, line=dict(color=C.COLOR_BEACON, dash="dot"), annotation_text="zweite Bake", annotation_position="top left")
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Wahrscheinlichkeit auf den beiden möglichen Orten", range=[0, 1.02], tickformat=".0%")
    return _base(fig, 300)


def build_ess_chart(a):
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.maximum(a.pf.ess, 1.0), mode="lines", line=dict(color=C.COLOR_PF, width=2), name="effektive Partikel"))
    r = a.pf.resampled
    fig.add_trace(go.Scatter(x=t[r], y=np.maximum(a.pf.ess[r], 1.0), mode="markers", marker=dict(color=C.COLOR_BEACON, size=7), name="neu gezogen"))
    fig.add_hline(y=a.settings.n, line=dict(color="#888", dash="dot"), annotation_text="alle Partikel", annotation_position="top left")
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Effektive Stichprobengröße (logarithmisch)", type="log")
    return _base(fig, 300)


def build_particle_sweep(res):
    rows = res["rows"]
    xs = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["rmse"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_PF, width=3), name="Partikelfilter"))
    fig.add_hline(y=res["ekf"], line=dict(color=C.COLOR_EKF, dash="dash"), annotation_text="EKF", annotation_position="top right")
    fig.add_hline(y=res["ukf"], line=dict(color=C.COLOR_UKF, dash="dash"), annotation_text="UKF", annotation_position="bottom right")
    fig.update_xaxes(title_text="Partikelzahl", type="log")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)


def build_time_chart(res):
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[r["seconds"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_PF, width=3), name="Rechenzeit je Fahrt"))
    fig.update_xaxes(title_text="Partikelzahl", type="log")
    fig.update_yaxes(title_text="Rechenzeit je Fahrt (s)", type="log")
    return _base(fig)


def build_ambiguity_chart(rows):
    labels = [("nie" if r["second"] == 0 else f"ab {r['second']} s") for r in rows]
    fig = go.Figure()
    for key, label, color in (("ekf", "EKF", C.COLOR_EKF), ("ukf", "UKF", C.COLOR_UKF), ("pf", "Partikelfilter", C.COLOR_PF)):
        fig.add_trace(go.Bar(x=labels, y=[r["found"][key] for r in rows], name=label, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Zweite Bake", type="category")
    fig.update_yaxes(title_text=f"Fahrten mit Endfehler unter {C.OK_ERR:.0f} m", range=[0, 1.02], tickformat=".0%")
    return _base(fig)


def build_dimension_chart(rows):
    fig = go.Figure()
    for ratio, color in zip(C.DIM_RATIOS, ("#4c78a8", "#c97b2a")):
        sub = [r for r in rows if r["ratio"] == ratio]
        name = f"Messrauschen = {ratio:g}× Prior-Streuung".replace(".", ",")
        fig.add_trace(go.Scatter(x=[r["d"] for r in sub], y=[max(r["mc"], 1e-4) for r in sub], mode="markers", marker=dict(color=color, size=9), name=f"gemessen, {name}"))
        fig.add_trace(go.Scatter(x=[r["d"] for r in sub], y=[max(r["formula"], 1e-4) for r in sub], mode="lines", line=dict(color=color, dash="dot"), name=f"Formel, {name}"))
    fig.update_xaxes(title_text="Dimension des Zustands", type="log")
    fig.update_yaxes(title_text="Anteil effektiver Partikel (logarithmisch)", type="log")
    return _base(fig)
