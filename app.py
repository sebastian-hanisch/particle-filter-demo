"""Partikelfilter: mehrgipflige Verteilungen mit Stichproben darstellen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Fahrzeug fährt an einer Bake vorbei, die nur die Entfernung misst: es kann auf zwei spiegelbildlichen Seiten der Fahrtachse sein.
Ein Gauß-Filter (EKF, UKF) kennt einen Gipfel; der Partikelfilter hält die Verteilung als Wolke von Stichproben und kann beide halten.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import pfl_constants as C
import pfl_evaluation as ev
from pfl_evaluation import Settings, analyse
from pfl_pdf_export import build_pdf
from pfl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pfl_visualization import build_ambiguity_chart, build_cloud_map, build_dimension_chart, build_ess_chart, build_mass_chart, build_particle_sweep, build_time_chart

st.set_page_config(page_title="Partikelfilter – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def second_label(v):
    return "nie" if v == 0 else f"ab Sekunde {v}"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_particles(settings):
    return ev.sweep_particles(settings)


@st.cache_data(show_spinner=False)
def _sweep_ambiguity(settings):
    return ev.sweep_ambiguity(settings)


@st.cache_data(show_spinner=False)
def _sweep_dimension(_key):
    return ev.sweep_dimension(None)


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (einige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner einige Sekunden)"):
        return fn(settings)


st.title("☁️ Partikelfilter: mehrgipflige Verteilungen mit Stichproben darstellen")
st.markdown(
    """
Eine Bake misst nur die **Entfernung** zum Fahrzeug. Dann kann es auf **zwei spiegelbildlichen Seiten** der Fahrtachse sein, mit genau derselben Entfernung: die Verteilung der möglichen Orte hat
**zwei Gipfel**. Ein Gauß-Filter (EKF, UKF) kennt nur einen und legt ihn **dazwischen**, wo das Fahrzeug nicht sein kann. Ein **Hidden-Markov-Modell** könnte beide Gipfel halten, braucht dafür aber einen
diskreten, kleinen Zustandsraum. Der **Partikelfilter** verbindet beides: kontinuierlicher, nichtlinearer Zustand wie beim EKF, mehrere Gipfel wie beim HMM. Er stellt die Verteilung als **Wolke von
Stichproben** dar, die vorhergesagt, nach der Messung gewichtet und bei Bedarf neu gezogen wird. Die Demo ist das sechste Stück der Linie **Zustandsschätzung und Filter**. Im Unterschied zu den Fall-Demos zeigt sie
ein einziges Verfahren an einem wachsenden Beispiel.

Weiter unten: drei Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Partikelfilter", expanded=True):
    st.markdown(
        """
Die Verteilung des Zustands wird durch **N Partikel** dargestellt, jeder ein möglicher Zustand mit einem Gewicht. Pro Sekunde:

1. **Vorhersagen:** jeder Partikel fährt mit dem Bewegungsmodell weiter, mit einer eigenen Zufallsbeschleunigung.
2. **Gewichten:** Partikel, deren vorhergesagte Messung zur echten passt, bekommen ein großes Gewicht, die anderen ein kleines.
3. **Neu ziehen:** Sind nur noch wenige Partikel wirklich beteiligt (**effektive Stichprobengröße** unter der Hälfte), werden N neue Partikel gezogen, bevorzugt aus den schweren.

Das Verfahren braucht **keine Linearisierung und keine Gauß-Annahme**: nur das Messmodell als Funktion, die man auswerten kann. Der Preis: Genauigkeit hängt an der **Partikelzahl**, und
sie muss mit der Dimension des Zustands stark wachsen (Experiment 3).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Messung**")
    mode = st.radio("Messart", C.MODES, key="mode_select", help="Nur Entfernung ist zweideutig (zwei Spiegelorte); mit Peilung ist die Lage eindeutig.")
    sigma_r = st.slider("Entfernungsrauschen (m)", *bounds("sr_slider"), key="sr_slider", step=0.5, help="Streuung der gemessenen Entfernung (Standardabweichung).")
    second = st.select_slider("Zweite Bake", C.SECOND_CHOICES, key="second_select", format_func=second_label,
                              help="Ab dieser Sekunde misst eine zweite Bake mit; sie löst die Mehrdeutigkeit auf. Nur im Modus „nur Entfernung“ wirksam.")
    st.markdown("**Das Verfahren**")
    n = st.select_slider("Partikelzahl", C.N_CHOICES, key="n_select", help="Mehr Partikel stellen die Verteilung genauer dar und kosten proportional mehr Rechenzeit.")
    q = st.slider("Fahrunruhe (m/s²)", *bounds("q_slider"), key="q_slider", step=0.02, help="Beschleunigungsrauschen der Fahrt; alle Filter kennen den wahren Wert.")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"mode_select": mode, "sr_slider": float(sigma_r), "n_select": int(n), "second_select": int(second), "q_slider": float(q), "seed_input": int(seed)})

settings = Settings(mode, float(sigma_r), int(n), int(second), float(q), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Endfehler EKF", f"{de(a.final['ekf'])} m", delta="gefunden" if a.found["ekf"] else f"verfehlt (über {C.OK_ERR:.0f} m)", delta_color="off")
m2.metric("Endfehler UKF", f"{de(a.final['ukf'])} m", delta="gefunden" if a.found["ukf"] else f"verfehlt (über {C.OK_ERR:.0f} m)", delta_color="off")
m3.metric("Endfehler Partikelfilter", f"{de(a.final['pf'])} m", delta="gefunden" if a.found["pf"] else f"verfehlt (über {C.OK_ERR:.0f} m)", delta_color="off")
m4.metric("Rechenzeit Partikelfilter", f"{a.pf.seconds:.2f} s".replace(".", ","), delta=f"{settings.n} Partikel, {C.T_STEPS} Sekunden", delta_color="off",
          help="Gemessene Zeit dieses Laufs auf dem Rechner, auf dem die Demo läuft.")

st.markdown("**Die Wolke zu einer Sekunde** – blau: Partikel, grüne Raute: wahre Position, grüne Umrandung: ihr Spiegelbild an der Fahrtachse, Quadrat/Dreieck/Kreis: Mittelwerte von EKF/UKF/Partikelfilter.")
t_show = st.slider("Zeitpunkt (s)", 0, C.T_STEPS - 1, min(C.T_STEPS - 1, 25), key="t_show", help="Welche Sekunde der Fahrt gezeigt wird.")
st.plotly_chart(build_cloud_map(a, int(t_show)), width="stretch", key="cloud_map")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Wahrscheinlichkeit auf den beiden möglichen Orten** (innerhalb 15 m)")
    st.plotly_chart(build_mass_chart(a), width="stretch", key="mass_chart")
with c2:
    st.markdown("**Wie viele Partikel zählen noch?**")
    st.plotly_chart(build_ess_chart(a), width="stretch", key="ess_chart")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="partikelfilter_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann lohnt sich die Stichprobenwolke?")
t_check = (settings.second - 1) if (settings.mode == "nur Entfernung" and settings.second > 0) else C.T_STEPS - 1
if settings.mode == "Entfernung und Peilung":
    st.info(
        f"**Eindeutige Messung:** Endfehler EKF {de(a.final['ekf'])} m, UKF {de(a.final['ukf'])} m, Partikelfilter {de(a.final['pf'])} m. Hier gibt es nur einen Gipfel; die Gauß-Filter reichen, "
        f"der Partikelfilter braucht {a.pf.seconds:.2f} s".replace(".", ",") + " und liefert dasselbe (Experiment 1: wie viele Partikel dafür nötig sind)."
    )
else:
    st.success(
        f"**Zweideutige Messung:** Vor dem Eingreifen der zweiten Bake (Sekunde {t_check}) liegt die Wahrscheinlichkeit auf den **beiden möglichen Orten** beim Partikelfilter bei "
        f"**{pct(a.pf.cand_mass[t_check])}**, beim UKF bei **{pct(a.cand_ukf[t_check])}**: der Gauß-Filter legt seine Masse dazwischen. Endfehler: EKF {de(a.final['ekf'])} m, "
        f"UKF {de(a.final['ukf'])} m, Partikelfilter {de(a.final['pf'])} m."
    )
st.caption("Eine einzelne Fahrt zeigt nur, was hier passiert ist; die Experimente unten mitteln über 30 feste Fahrten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie viele Partikel braucht man bei eindeutiger Messung?")
st.caption(f"Messart „Entfernung und Peilung“, Median über {len(C.SWEEP_SEEDS)} feste Fahrten; Rauschen und Fahrunruhe wie in der Seitenleiste. Gestrichelt: EKF und UKF.")
res = run_experiment("particles", _sweep_particles, settings, f"▶ Experiment 1 rechnen ({(len(C.N_CHOICES) + 1) * len(C.SWEEP_SEEDS)} Fahrten)")
if res is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_particle_sweep(res), width="stretch", key="particle_chart")
    with c2:
        st.plotly_chart(build_time_chart(res), width="stretch", key="time_chart")
    rows = {r["n"]: r for r in res["rows"]}
    st.info(
        f"Mit {C.N_CHOICES[0]} Partikeln liegt der Fehler bei {de(rows[C.N_CHOICES[0]]['rmse'])} m, mit 1000 bei {de(rows[1000]['rmse'], 2)} m und mit {C.N_CHOICES[-1]} bei {de(rows[C.N_CHOICES[-1]]['rmse'], 2)} m; "
        f"EKF {de(res['ekf'], 2)} m, UKF {de(res['ukf'], 2)} m. Ab etwa 1000 Partikeln ist der Partikelfilter hier auf dem Niveau der Gauß-Filter, bei etwa dem {rows[C.N_CHOICES[-1]]['seconds'] / rows[1000]['seconds']:.0f}-Fachen der Rechenzeit "
        f"({de(rows[1000]['seconds'], 3)} s gegen {de(rows[C.N_CHOICES[-1]]['seconds'], 3)} s je Fahrt auf diesem Rechner)."
    )

st.subheader("🔬 Experiment 2: Mehrdeutigkeit – wer findet die richtige Seite?")
st.caption(
    f"Messart „nur Entfernung“, {len(C.SWEEP_SEEDS)} feste Fahrten, Partikelzahl und Rauschen wie in der Seitenleiste. „Gefunden“: Endfehler unter {C.OK_ERR:.0f} m. Die zweite Bake löst die "
    "Mehrdeutigkeit zu dem angegebenen Zeitpunkt auf; „nie“ heißt: sie bleibt bis zum Ende."
)
amb = run_experiment("ambiguity", _sweep_ambiguity, settings, f"▶ Experiment 2 rechnen ({len(C.SECOND_CHOICES) * len(C.SWEEP_SEEDS)} Fahrten)")
if amb is not None:
    st.plotly_chart(build_ambiguity_chart(amb), width="stretch", key="ambiguity_chart")
    r30 = next(r for r in amb if r["second"] == 30)
    rnever = next(r for r in amb if r["second"] == 0)
    st.info(
        f"Mit zweiter Bake ab Sekunde 30 finden UKF {pct(r30['found']['ukf'])} und Partikelfilter {pct(r30['found']['pf'])} der Fahrten die richtige Seite, der EKF {pct(r30['found']['ekf'])}. "
        f"**Beim Positionsfehler ist der Partikelfilter hier nicht besser als der UKF.** Sein Vorteil liegt in der Darstellung vor der zweiten Bake: auf den beiden möglichen Orten liegen "
        f"{pct(r30['pf_cand'])} seiner Wahrscheinlichkeit gegen {pct(r30['ukf_cand'])} beim UKF. Kommt die zweite Bake nie, findet keiner die richtige Seite zuverlässig "
        f"(Partikelfilter {pct(rnever['found']['pf'])}, UKF {pct(rnever['found']['ukf'])}); der Partikelfilter hält aber beide Möglichkeiten offen."
    )

st.subheader("🔬 Experiment 3: Was kostet eine höhere Dimension?")
st.caption(
    f"Reine Rechenprobe ohne Fahrzeug: {C.DIM_PARTICLES} Partikel aus einem Gauß-Prior in d Dimensionen, jede Komponente wird einzeln mit dem Rauschen gemessen, das im Diagramm steht. "
    f"Gezeigt ist der Anteil effektiver Partikel nach **einem** Gewichtungsschritt (Mittel über {C.DIM_REPEATS} Wiederholungen), gemessen und nach geschlossener Formel."
)
dim = run_experiment("dimension", lambda s: _sweep_dimension(0), settings, f"▶ Experiment 3 rechnen ({len(C.DIM_SWEEP) * len(C.DIM_RATIOS) * C.DIM_REPEATS} Wiederholungen)")
if dim is not None:
    st.plotly_chart(build_dimension_chart(dim), width="stretch", key="dimension_chart")
    hi = [r for r in dim if r["ratio"] == C.DIM_RATIOS[1]]
    d8 = next(r for r in hi if r["d"] == 8)
    d32 = next(r for r in hi if r["d"] == 32)
    st.info(
        f"Bei scharfer Messung (Rauschen {de(C.DIM_RATIOS[1])}× der Prior-Streuung) bleiben von {C.DIM_PARTICLES} Partikeln in 8 Dimensionen im Mittel {C.DIM_PARTICLES * d8['mc']:.0f} wirksam, "
        f"in 32 Dimensionen {C.DIM_PARTICLES * d32['mc']:.1f}: praktisch ein einziger Partikel trägt die ganze Verteilung (**Gewichtskollaps**). Die Formel stimmt mit der Messung überein, "
        "solange noch genug Partikel übrig sind; bei den kleinen Werten liegt die Messung wegen der endlichen Stichprobe darüber."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Partikel für die Aufgabe** | Mit 300 oder 1000 Partikeln geht in manchen Fahrten der richtige Gipfel verloren, bevor die zweite Bake ihn bräuchte (Experiment 2). Die Verteilung wird von zufälligem Neuziehen dann nicht mehr abgebildet. | Mehr Partikel, bessere Vorschlagsverteilungen |
| **Niedrige Dimension** | Der Bedarf wächst mit der Dimension des Zustands sehr schnell (Experiment 3). Für Zustände mit Dutzenden oder mehr Dimensionen ist der einfache Partikelfilter keine Lösung. | Verfahren für hohe Dimension (Ensemble-Filter) |
| **Eine Gipfelverteilung zeigt sich als Stichprobe** | Eine **Schätzung** (Mittelwert) zwischen den Gipfeln ist trotzdem falsch: bei zwei Gipfeln ist der Mittelwert kein sinnvoller Ort. Man muss die Wolke selbst lesen (Modus, Gipfelmasse). | – |
| **Das Modell stimmt** | Der Partikelfilter ist so gut wie Bewegungs- und Messmodell. Falsches Rauschen macht ihn genauso überselbstsicher wie die anderen Filter. | Konsistenzprüfung |
| **Rechenzeit ist kein Problem** | Kosten wachsen proportional zur Partikelzahl; Echtzeit auf kleinen Geräten kann schwierig werden. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ziel.** Verteilung des Zustands $x_k$ gegeben alle Messungen $z_{1:k}$, $p(x_k \mid z_{1:k})$, als gewichtete Stichprobe $\{(x_k^{(i)}, w_k^{(i)})\}_{i=1}^N$.

**Bootstrap-Partikelfilter.** Pro Schritt: (1) $x_k^{(i)} \sim p(x_k \mid x_{k-1}^{(i)})$ (hier $x_k = F x_{k-1} + w$, $w \sim \mathcal N(0, Q)$); (2) $w_k^{(i)} \propto w_{k-1}^{(i)}\,p(z_k \mid x_k^{(i)})$,
hier $p(z \mid x) \propto \exp(-\tfrac12 (z - h(x))^\top R^{-1} (z - h(x)))$; (3) Resampling, wenn die effektive Stichprobengröße
$$N_{\mathrm{eff}} = 1 \Big/ \textstyle\sum_i (w^{(i)})^2$$
unter $N/2$ fällt (systematisches Resampling: ein Zufallswert, $N$ gleichmäßig verteilte Stützpunkte auf der Summenfolge der Gewichte).

**Schätzer.** $\hat x_k = \sum_i w^{(i)} x^{(i)}$ (Mittelwert); bei mehreren Gipfeln ist der Mittelwert kein Ort der Verteilung. Die Demo liest deshalb die **Masse** innerhalb eines Radius um die beiden möglichen Orte.

**Gewichtskollaps in der Dimension.** Prior $\mathcal N(0, I_d)$, Messung jeder Komponente mit Streuung $s$, Messwert gleich dem Prior-Mittel: $w(x) \propto \exp(-\lVert x\rVert^2 / 2s^2)$ und
$$\frac{N_{\mathrm{eff}}}{N} \approx \frac{(E\,w)^2}{E\,w^2} = \left(\frac{1 + 2/s^2}{(1 + 1/s^2)^2}\right)^{d/2}.$$
Dieser Anteil fällt exponentiell mit $d$.

**Literatur.** N. J. Gordon, D. J. Salmond, A. F. M. Smith (1993), *Novel approach to nonlinear/non-Gaussian Bayesian state estimation*, IEE Proceedings F (Radar and Signal Processing) 140(2), 107–113.
C. Snyder, T. Bengtsson, P. Bickel, J. Anderson (2008), *Obstacles to high-dimensional particle filtering*, Monthly Weather Review 136(12), 4629–4640 (Bedarf an Partikeln wächst beim einfachen Filter exponentiell mit der Dimension; Resampling behebt das nicht).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
