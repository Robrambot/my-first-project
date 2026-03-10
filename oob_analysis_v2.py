import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# CONFIG
# =========================================================
signals = [
    "InverterMotorCurrent",
    "InverterMotorPower",
    "InverterMotorTorque",
    "Drive_SpeedSensorShaftToBasket"
]

MAX_MASS_LB = 4
MAX_ITER = 3
OUTPUT_DIR = "oob_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Excluir masas si hace falta
EXCLUDED_MASSES = {0}   # por ahora no usar 0 lb hasta tener CSV correcto

# Detección de onset real por velocidad
ONSET_SPEED_THRESHOLD = 5.0        # rpm
ONSET_CONFIRM_WINDOW = 20          # muestras para confirmar
ONSET_CONFIRM_RATIO = 0.70         # 70% de la ventana por arriba del umbral

# Región estable para marcar en gráficas
STABLE_SPEED_TARGET = 670.0
STABLE_SPEED_TOL = 10.0

# Colores por masa
MASS_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
}

# Línea por iteración
ITER_DASH = {
    1: "solid",
    2: "dash",
    3: "dot"
}

# Símbolo por iteración
ITER_SYMBOL = {
    1: "circle",
    2: "square",
    3: "diamond"
}

# =========================================================
# HELPERS
# =========================================================
def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))

def peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(x) - np.min(x))

def detect_and_load_csv(filepath: str, required_cols: list[str]) -> pd.DataFrame:
    """
    Carga robusta:
    1) lectura normal
    2) si no encuentra columnas, reintenta con skiprows=7
    """
    candidates = []

    try:
        df = pd.read_csv(filepath)
        candidates.append(("normal", df))
    except Exception:
        pass

    try:
        df = pd.read_csv(filepath, skiprows=7)
        candidates.append(("skiprows7", df))
    except Exception:
        pass

    for mode, df in candidates:
        if all(col in df.columns for col in required_cols):
            df = df[required_cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
            if len(df) > 10:
                print(f"[OK] {os.path.basename(filepath)} cargado con modo: {mode}")
                return df

    raise ValueError(f"No se pudieron encontrar las columnas requeridas en {filepath}")

def parse_mass_iter(filename: str):
    """
    Espera nombres tipo:
    1lb_it1.csv
    2lb_it3.csv
    """
    base = os.path.basename(filename)
    match = re.match(r"(?i)^(\d+)lb_it(\d+)\.csv$", base)
    if not match:
        return None
    mass = int(match.group(1))
    iteration = int(match.group(2))
    return mass, iteration

def safe_sort_key(key_tuple):
    mass, iteration = key_tuple
    return (mass, iteration)

def detect_motion_onset(speed: np.ndarray,
                        threshold: float = ONSET_SPEED_THRESHOLD,
                        confirm_window: int = ONSET_CONFIRM_WINDOW,
                        confirm_ratio: float = ONSET_CONFIRM_RATIO):
    """
    Detecta onset real de movimiento usando velocidad.
    Regla:
    - primer índice donde speed >= threshold
    - y en las siguientes confirm_window muestras al menos confirm_ratio
      permanezcan por arriba del umbral
    """
    speed = np.asarray(speed, dtype=float)
    n = len(speed)

    if n < confirm_window + 2:
        return None, False, "signal too short"

    above = speed >= threshold

    for i in range(n - confirm_window):
        if above[i]:
            window = above[i:i + confirm_window]
            if np.mean(window) >= confirm_ratio:
                return int(i), True, "ok"

    # fallback: si nunca cumple, usar primer punto arriba del umbral
    idx = np.where(above)[0]
    if len(idx) > 0:
        return int(idx[0]), True, "fallback_first_above_threshold"

    return None, False, "no onset detected"

def detect_stable_region(speed: np.ndarray,
                         target: float = STABLE_SPEED_TARGET,
                         tol: float = STABLE_SPEED_TOL):
    """
    Detecta la región estable donde la velocidad está dentro de target ± tol.
    Si hay varias regiones, toma la más larga.
    """
    speed = np.asarray(speed, dtype=float)
    mask = (speed >= (target - tol)) & (speed <= (target + tol))

    if not np.any(mask):
        return None, None, False

    idx = np.where(mask)[0]
    segments = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))

    seg = max(segments, key=lambda x: x[1] - x[0] + 1)
    return int(seg[0]), int(seg[1]), True

def crop_all_to_common_length(dfs: dict) -> tuple[dict, int]:
    min_len = min(len(v["df"]) for v in dfs.values())
    out = {}
    for key, item in dfs.items():
        out[key] = item.copy()
        out[key]["df"] = item["df"].iloc[:min_len].reset_index(drop=True)
    return out, min_len

def build_shapes_for_stable_regions(aligned_dict: dict):
    shapes = []

    for key in sorted(aligned_dict.keys(), key=safe_sort_key):
        item = aligned_dict[key]
        df = item["df"]
        mass = item["mass_lb"]

        speed = df["Drive_SpeedSensorShaftToBasket"].values
        start_idx, end_idx, found = detect_stable_region(speed)

        if found:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=int(start_idx),
                    x1=int(end_idx),
                    y0=0,
                    y1=1,
                    fillcolor=MASS_COLORS.get(mass, "#999999"),
                    opacity=0.08,
                    layer="below",
                    line_width=0
                )
            )

    return shapes

# =========================================================
# DISCOVER FILES
# =========================================================
all_csv = glob.glob("*.csv")
parsed_files = []

for f in all_csv:
    parsed = parse_mass_iter(f)
    if parsed is None:
        continue
    mass, iteration = parsed
    if mass in EXCLUDED_MASSES:
        continue
    if mass <= MAX_MASS_LB and iteration <= MAX_ITER:
        parsed_files.append((mass, iteration, f))

if not parsed_files:
    raise FileNotFoundError(
        "No se encontraron archivos válidos con patrón tipo 1lb_it1.csv, 2lb_it2.csv, etc."
    )

parsed_files = sorted(parsed_files, key=lambda x: (x[0], x[1]))

print("\n=== Archivos detectados ===")
for mass, iteration, f in parsed_files:
    print(f"  {f} -> mass={mass} lb, iter={iteration}")

# =========================================================
# LOAD DATA
# =========================================================
data_raw = {}
skipped = []

for mass, iteration, filepath in parsed_files:
    try:
        df = detect_and_load_csv(filepath, signals)
        data_raw[(mass, iteration)] = {
            "file": filepath,
            "mass_lb": mass,
            "iteration": iteration,
            "df": df
        }
    except Exception as e:
        skipped.append((filepath, str(e)))
        warnings.warn(f"Se omitió {filepath}: {e}")

if not data_raw:
    raise RuntimeError("No hubo ningún archivo válido para analizar.")

if skipped:
    print("\n=== Archivos omitidos ===")
    for f, msg in skipped:
        print(f"  {f}: {msg}")

# =========================================================
# DETECT ONSET AND ALIGN BY REAL MOTION ONSET
# =========================================================
onset_info = {}
valid_after_onset = {}
invalid_onset = []

for key, item in data_raw.items():
    df = item["df"].copy()
    speed = df["Drive_SpeedSensorShaftToBasket"].values

    onset_idx, found, note = detect_motion_onset(speed)

    if not found or onset_idx is None:
        invalid_onset.append((item["file"], note))
        continue

    onset_info[key] = {
        "onset_idx_raw": onset_idx,
        "onset_note": note
    }

if not onset_info:
    raise RuntimeError("No se detectó onset válido en ningún archivo.")

anchor_onset = min(v["onset_idx_raw"] for v in onset_info.values())

for key, item in data_raw.items():
    if key not in onset_info:
        continue

    df = item["df"].copy()
    onset_idx = onset_info[key]["onset_idx_raw"]

    shift = onset_idx - anchor_onset
    if shift > 0:
        df_al = df.iloc[shift:].reset_index(drop=True)
    else:
        df_al = df.reset_index(drop=True)

    valid_after_onset[key] = {
        "file": item["file"],
        "mass_lb": item["mass_lb"],
        "iteration": item["iteration"],
        "df": df_al,
        "Onset_Index_Raw": onset_idx,
        "Onset_Shift_Applied": max(shift, 0),
        "Onset_Index_Aligned": onset_idx - max(shift, 0),
        "Onset_Note": onset_info[key]["onset_note"]
    }

print(f"\nÍndice ancla común de onset: {anchor_onset}")

print("\n=== Onset detectado por archivo ===")
for key in sorted(valid_after_onset.keys(), key=safe_sort_key):
    item = valid_after_onset[key]
    print(
        f"  {os.path.basename(item['file'])}: "
        f"onset_raw={item['Onset_Index_Raw']}, "
        f"shift_applied={item['Onset_Shift_Applied']}, "
        f"onset_aligned={item['Onset_Index_Aligned']}, "
        f"note={item['Onset_Note']}"
    )

if invalid_onset:
    print("\n=== Archivos con onset no válido ===")
    for f, note in invalid_onset:
        print(f"  {f}: {note}")

# =========================================================
# COMMON LENGTH AFTER ONSET ALIGNMENT
# =========================================================
aligned, common_len = crop_all_to_common_length(valid_after_onset)
print(f"\nLongitud común usada en todos los análisis: {common_len} samples")

# =========================================================
# METRICS TABLE
# =========================================================
rows = []

for key in sorted(aligned.keys(), key=safe_sort_key):
    item = aligned[key]
    df = item["df"]

    curr = df["InverterMotorCurrent"].values
    power = df["InverterMotorPower"].values
    torque = df["InverterMotorTorque"].values
    speed = df["Drive_SpeedSensorShaftToBasket"].values

    stable_start, stable_end, stable_found = detect_stable_region(speed)

    row = {
        "file": os.path.basename(item["file"]),
        "mass_lb": item["mass_lb"],
        "iteration": item["iteration"],
        "n_samples_used": len(df),
        "Onset_Index_Raw": item["Onset_Index_Raw"],
        "Onset_Shift_Applied": item["Onset_Shift_Applied"],
        "Onset_Index_Aligned": item["Onset_Index_Aligned"],
        "Onset_Note": item["Onset_Note"],
        "Stable_Region_Found": stable_found,
        "Stable_Start": stable_start if stable_found else np.nan,
        "Stable_End": stable_end if stable_found else np.nan,
        "Mean_Current": float(np.mean(curr)),
        "Mean_Power": float(np.mean(power)),
        "Mean_Torque": float(np.mean(torque)),
        "Mean_Speed": float(np.mean(speed)),
        "RMS_Current": rms(curr),
        "RMS_Power": rms(power),
        "RMS_Torque": rms(torque),
        "RMS_Speed": rms(speed),
        "Envelope_Torque_PkPk": peak_to_peak(torque),
        "OOB_Metric": rms(curr) + rms(torque),
    }
    rows.append(row)

metrics_df = pd.DataFrame(rows).sort_values(["mass_lb", "iteration"]).reset_index(drop=True)

print("\n=== Tabla de métricas por iteración ===")
print(metrics_df.to_string(index=False))

metrics_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nTabla guardada en: {metrics_csv}")

# =========================================================
# INTERACTIVE SIGNAL PLOTS
# =========================================================
def build_signal_figure(signal_name: str, aligned_dict: dict, output_html: str):
    fig = go.Figure()
    trace_meta = []

    sorted_keys = sorted(aligned_dict.keys(), key=safe_sort_key)

    for key in sorted_keys:
        item = aligned_dict[key]
        df = item["df"]
        mass = item["mass_lb"]
        iteration = item["iteration"]

        x = np.arange(len(df))
        y = df[signal_name].values

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{mass}lb_it{iteration}",
            line=dict(
                color=MASS_COLORS.get(mass, None),
                dash=ITER_DASH.get(iteration, "solid"),
                width=2
            ),
            hovertemplate=(
                f"Trace: {mass}lb_it{iteration}<br>"
                "Sample: %{x}<br>"
                "Value: %{y:.5f}<extra></extra>"
            )
        ))
        trace_meta.append({"mass": mass, "iteration": iteration})

    unique_masses = sorted(set(item["mass_lb"] for item in aligned_dict.values()))

    buttons = []
    buttons.append(dict(
        label="All masses",
        method="update",
        args=[{"visible": [True] * len(trace_meta)},
              {"title": f"{signal_name} - All masses"}]
    ))

    for m in unique_masses:
        visible = [meta["mass"] == m for meta in trace_meta]
        buttons.append(dict(
            label=f"{m} lb",
            method="update",
            args=[{"visible": visible},
                  {"title": f"{signal_name} - {m} lb"}]
        ))

    shapes = build_shapes_for_stable_regions(aligned_dict)

    fig.update_layout(
        title=f"{signal_name} - All masses",
        xaxis_title="Sample from motion onset",
        yaxis_title=signal_name,
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top"
        )],
        legend=dict(
            title="Mass / Iteration",
            orientation="v"
        ),
        shapes=shapes
    )

    fig.add_vline(
        x=anchor_onset,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text="Motion onset anchor",
        annotation_position="top left"
    )

    fig.write_html(output_html)
    print(f"Gráfica guardada: {output_html}")

for sig in signals:
    out_html = os.path.join(OUTPUT_DIR, f"{sig}_interactive.html")
    build_signal_figure(sig, aligned, out_html)

# =========================================================
# RMS SUBPLOTS
# =========================================================
fig_rms = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "RMS Current",
        "RMS Power",
        "RMS Torque",
        "RMS Speed"
    ]
)

rms_config = [
    ("RMS_Current", 1, 1),
    ("RMS_Power", 1, 2),
    ("RMS_Torque", 2, 1),
    ("RMS_Speed", 2, 2),
]

unique_masses = sorted(metrics_df["mass_lb"].unique())

for metric_name, row, col in rms_config:
    for mass in unique_masses:
        sub = metrics_df[metrics_df["mass_lb"] == mass].sort_values("iteration")
        if sub.empty:
            continue

        symbols = [ITER_SYMBOL.get(int(it), "circle") for it in sub["iteration"].tolist()]

        fig_rms.add_trace(
            go.Scatter(
                x=sub["iteration"],
                y=sub[metric_name],
                mode="lines+markers+text",
                name=f"{mass} lb",
                marker=dict(
                    size=11,
                    symbol=symbols,
                    color=MASS_COLORS.get(mass, None)
                ),
                line=dict(color=MASS_COLORS.get(mass, None), width=2),
                text=[f"it{int(it)}" for it in sub["iteration"]],
                textposition="top center",
                legendgroup=f"{mass}lb",
                showlegend=(metric_name == "RMS_Current"),
                hovertemplate=(
                    f"Mass: {mass} lb<br>"
                    "Iteration: %{x}<br>"
                    f"{metric_name}: "+"%{y:.5f}<extra></extra>"
                )
            ),
            row=row,
            col=col
        )

fig_rms.update_xaxes(title_text="Iteration", dtick=1)
fig_rms.update_layout(
    title="RMS Summary by Iteration",
    template="plotly_white",
    height=800,
    width=1100
)

rms_html = os.path.join(OUTPUT_DIR, "RMS_summary.html")
fig_rms.write_html(rms_html)
print(f"Gráfica guardada: {rms_html}")

# =========================================================
# OOB METRIC PLOT
# =========================================================
fig_oob = go.Figure()

for mass in unique_masses:
    sub = metrics_df[metrics_df["mass_lb"] == mass].sort_values("iteration")
    if sub.empty:
        continue

    symbols = [ITER_SYMBOL.get(int(it), "circle") for it in sub["iteration"].tolist()]

    fig_oob.add_trace(go.Scatter(
        x=sub["iteration"],
        y=sub["OOB_Metric"],
        mode="lines+markers+text",
        name=f"{mass} lb",
        marker=dict(
            size=11,
            symbol=symbols,
            color=MASS_COLORS.get(mass, None)
        ),
        line=dict(color=MASS_COLORS.get(mass, None), width=2),
        text=[f"it{int(it)}" for it in sub["iteration"]],
        textposition="top center",
        hovertemplate=(
            f"Mass: {mass} lb<br>"
            "Iteration: %{x}<br>"
            "OOB Metric: %{y:.5f}<extra></extra>"
        )
    ))

fig_oob.update_layout(
    title="OOB Metric by Iteration",
    xaxis_title="Iteration",
    yaxis_title="OOB_Metric = RMS_Current + RMS_Torque",
    template="plotly_white"
)

oob_html = os.path.join(OUTPUT_DIR, "OOB_metric.html")
fig_oob.write_html(oob_html)
print(f"Gráfica guardada: {oob_html}")

# =========================================================
# MASS SUMMARY
# =========================================================
mass_summary = (
    metrics_df.groupby("mass_lb", as_index=False)
    .agg({
        "Mean_Current": ["mean", "std"],
        "Mean_Power": ["mean", "std"],
        "Mean_Torque": ["mean", "std"],
        "Mean_Speed": ["mean", "std"],
        "RMS_Current": ["mean", "std"],
        "RMS_Power": ["mean", "std"],
        "RMS_Torque": ["mean", "std"],
        "RMS_Speed": ["mean", "std"],
        "Envelope_Torque_PkPk": ["mean", "std"],
        "OOB_Metric": ["mean", "std"]
    })
)

mass_summary.columns = [
    "mass_lb",
    "Mean_Current_mean", "Mean_Current_std",
    "Mean_Power_mean", "Mean_Power_std",
    "Mean_Torque_mean", "Mean_Torque_std",
    "Mean_Speed_mean", "Mean_Speed_std",
    "RMS_Current_mean", "RMS_Current_std",
    "RMS_Power_mean", "RMS_Power_std",
    "RMS_Torque_mean", "RMS_Torque_std",
    "RMS_Speed_mean", "RMS_Speed_std",
    "Envelope_Torque_PkPk_mean", "Envelope_Torque_PkPk_std",
    "OOB_Metric_mean", "OOB_Metric_std"
]

mass_summary_csv = os.path.join(OUTPUT_DIR, "metrics_mass_summary.csv")
mass_summary.to_csv(mass_summary_csv, index=False)

print("\n=== Resumen por masa ===")
print(mass_summary.to_string(index=False))
print(f"\nResumen por masa guardado en: {mass_summary_csv}")

print("\nProceso terminado.")
print(f"Revisa la carpeta: {OUTPUT_DIR}")
print("Abre los archivos .html en tu navegador para ver las gráficas interactivas.")