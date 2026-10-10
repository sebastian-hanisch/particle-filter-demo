"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Zeitachse: ein Schritt = eine Sekunde, 70 Schritte
DT = 1.0
T_STEPS = 70
N_BURN = 5

# Fahrt: Straße bei y = 60 m, die erste Bake steht im Ursprung, die zweite (optional) bei (0, 30)
START = (-100.0, 60.0)
SPEED = 5.0
BEACON_1 = (0.0, 0.0)
BEACON_2 = (0.0, 30.0)
X_PRIOR_STD = 5.0                 # Vorwissen über x: eng
Y_PRIOR_STD = 60.0                # Vorwissen über y (Straßenseite unbekannt): weit, symmetrisch um 0
V_PRIOR_STD = 1.0
OK_ERR = 15.0                     # „gefunden“: Endfehler unter dieser Grenze (m)
PARTICLE_SHOW = 1500              # so viele Partikel je Zeitpunkt werden für die Darstellung aufgehoben

# Reglergrenzen
MODES = ("Entfernung und Peilung", "nur Entfernung")
DEFAULT_MODE = "nur Entfernung"
SIGMA_R_MIN, SIGMA_R_MAX, DEFAULT_SIGMA_R = 1.0, 10.0, 3.0
SIGMA_B_DEG = 3.0                 # Peilrauschen im Modus „Entfernung und Peilung“ (fest)
N_CHOICES = (100, 300, 1000, 3000, 10000)
DEFAULT_N = 1000
SECOND_CHOICES = (20, 30, 40, 50, 0)     # Sekunde, ab der die zweite Bake misst; 0 = nie
DEFAULT_SECOND = 30
Q_MIN, Q_MAX, DEFAULT_Q = 0.02, 0.5, 0.05
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100030))
DIM_SWEEP = (1, 2, 4, 8, 16, 32)
DIM_PARTICLES = 1000
DIM_RATIOS = (1.0, 0.3)           # Messrauschen im Verhältnis zur Prior-Streuung je Komponente
DIM_REPEATS = 200

COLOR_TRUTH = "#2F6B65"
COLOR_EKF = "#7a7a7a"
COLOR_UKF = "#c97b2a"
COLOR_PF = "#4c78a8"
COLOR_PARTICLE = "rgba(76,120,168,0.35)"
COLOR_BEACON = "#b04a4a"

PRESETS = {
    "Standard (Voreinstellung)": {"mode": DEFAULT_MODE, "sr": DEFAULT_SIGMA_R, "n": DEFAULT_N, "second": DEFAULT_SECOND, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Mehrdeutigkeit bleibt": {"mode": "nur Entfernung", "sr": DEFAULT_SIGMA_R, "n": DEFAULT_N, "second": 0, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Eindeutige Messung": {"mode": "Entfernung und Peilung", "sr": DEFAULT_SIGMA_R, "n": 3000, "second": 0, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Zu wenige Partikel": {"mode": "nur Entfernung", "sr": DEFAULT_SIGMA_R, "n": 100, "second": DEFAULT_SECOND, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "Eine Bake misst nur die Entfernung: das Fahrzeug kann auf zwei spiegelbildlichen Straßenseiten sein. Ab Sekunde 30 kommt eine zweite Bake dazu.",
    "Mehrdeutigkeit bleibt": "Auch eine zweite Bake kommt nie: die Verteilung bleibt zweigipflig bis zum Ende.",
    "Eindeutige Messung": "Entfernung und Peilung: nur ein Gipfel. Hier braucht es keinen Partikelfilter, die Frage ist, was er kostet.",
    "Zu wenige Partikel": "100 Partikel genügen nicht, um beide Gipfel und die Fahrt sicher zu halten.",
}
