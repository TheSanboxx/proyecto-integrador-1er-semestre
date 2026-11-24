"""Aplicación principal en Streamlit para el Proyecto Integrador de 1er Semestre.

Eje: De los Datos a la Historia — Construcción de una Aplicación Interactiva
de Exploración de Datos.
Autores: Daniel C. - Santiago B.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================================
# src/data_loader.py — Carga, validacion y limpieza del dataset
# ==========================================================================

DATASET_FILENAME = "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"

EXPECTED_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
    "addiction_level",
    "addicted_label",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "addicted_label",
)

STRESS_LABELS_ES: dict[str, str] = {
    "Low": "Bajo",
    "Medium": "Medio",
    "High": "Alto",
}

ADDICTION_LABELS_ES: dict[str, str] = {
    "None": "Sin adicción",
    "Mild": "Leve",
    "Moderate": "Moderada",
    "Severe": "Severa",
}

GENDER_LABELS_ES: dict[str, str] = {
    "Female": "Femenino",
    "Male": "Masculino",
    "Other": "Otro",
}

IMPACT_LABELS_ES: dict[str, str] = {
    "Yes": "Sí",
    "No": "No",
}

STRESS_ORDER: list[str] = ["Low", "Medium", "High"]
ADDICTION_ORDER: list[str] = ["None", "Mild", "Moderate", "Severe"]
SCREEN_BRACKET_ORDER: list[str] = [
    "Moderado (< 5h)",
    "Alto (5 - 8h)",
    "Crítico (> 8h)",
]
AGE_GROUP_ORDER: list[str] = [
    "18 - 25 (Jóvenes)",
    "26 - 35 (Adultos jóvenes)",
]

VALID_CATEGORIES: dict[str, set[str]] = {
    "gender": set(GENDER_LABELS_ES.keys()),
    "stress_level": set(STRESS_LABELS_ES.keys()),
    "addiction_level": set(ADDICTION_LABELS_ES.keys()),
    "academic_work_impact": set(IMPACT_LABELS_ES.keys()),
}


def resolve_dataset_path(custom_path: str | Path | None = None) -> Path:
    """Encuentra la ruta del archivo CSV en la raíz o en la carpeta data/."""
    if custom_path is not None:
        candidate = Path(custom_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {candidate}")

    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / DATASET_FILENAME,
        project_root / "data" / DATASET_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No se encontró '{DATASET_FILENAME}' en la raíz del proyecto ni en 'data/'."
    )


def classify_screen_bracket(hours: float) -> str:
    """Clasifica el tiempo diario de pantalla en tres tramos de riesgo."""
    if hours < 5.0:
        return "Moderado (< 5h)"
    elif hours <= 8.0:
        return "Alto (5 - 8h)"
    else:
        return "Crítico (> 8h)"


def classify_age_group(age: int) -> str:
    """Agrupa la edad destacando a la población joven (18 a 25 años)."""
    if 18 <= age <= 25:
        return "18 - 25 (Jóvenes)"
    else:
        return "26 - 35 (Adultos jóvenes)"


def classify_sleep_status(hours: float) -> str:
    """Clasifica las horas de sueño diarias según criterios básicos de descanso."""
    if hours < 6.0:
        return "Déficit (< 6h)"
    elif hours <= 7.0:
        return "Aceptable (6 - 7h)"
    else:
        return "Saludable (> 7h)"


def _map_stress_score(level: str) -> int:
    """Convierte el nivel de estrés en una escala numérica ordinal (0, 1, 2)."""
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    return mapping.get(level, 0)


def load_and_clean_data(
    csv_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Carga el CSV real, aplica limpieza estructurada y genera variables derivadas.

    Nota técnica: En pandas, el texto literal 'None' del archivo original suele
    interpretarse como NaN por defecto (819 filas en 'addiction_level'). Aquí
    preservamos explícitamente 'None' como categoría válida ('Sin adicción').
    """
    resolved_path = resolve_dataset_path(csv_path)

    # Leemos primero con la configuración por defecto para cuantificar cuántos
    # valores interpretaba pandas como nulos (como en Google Colab), y luego
    # cargamos preservando 'None' como texto válido.
    raw_default_df = pd.read_csv(resolved_path)
    default_nulls_addiction = int(raw_default_df["addiction_level"].isna().sum())

    df = pd.read_csv(resolved_path, keep_default_na=False)
    filas_iniciales = int(len(df))

    missing_columns: list[str] = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise ValueError(f"Faltan columnas esperadas en el CSV: {missing_columns}")

    # Limpieza de espacios y normalización de la columna addiction_level
    for col in VALID_CATEGORIES:
        df[col] = df[col].astype(str).str.strip()

    df["addiction_level"] = df["addiction_level"].replace({"": "None", "nan": "None", "NaN": "None"})

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    filas_antes_duplicados = len(df)
    df = df.drop_duplicates(subset=["transaction_id"]).copy()
    duplicados_eliminados = int(filas_antes_duplicados - len(df))

    # Verificación de categorías válidas usando conjuntos (sets)
    invalid_summary: dict[str, list[str]] = {}
    for col, expected_set in VALID_CATEGORIES.items():
        observed_set = set(df[col].unique())
        unexpected = sorted(observed_set - expected_set)
        if unexpected:
            invalid_summary[col] = unexpected

    # Columnas derivadas usando bucles y condicionales
    screen_brackets: list[str] = []
    age_groups: list[str] = []
    sleep_statuses: list[str] = []
    stress_scores: list[int] = []
    impact_binaries: list[int] = []

    for row in df.itertuples(index=False):
        screen_brackets.append(classify_screen_bracket(float(row.daily_screen_time_hours)))
        age_groups.append(classify_age_group(int(row.age)))
        sleep_statuses.append(classify_sleep_status(float(row.sleep_hours)))
        stress_scores.append(_map_stress_score(str(row.stress_level)))
        impact_binaries.append(1 if str(row.academic_work_impact) == "Yes" else 0)

    df["screen_bracket"] = screen_brackets
    df["age_group"] = age_groups
    df["sleep_status"] = sleep_statuses
    df["stress_score"] = stress_scores
    df["impact_binary"] = impact_binaries
    df["weekend_delta"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]

    # Etiquetas legibles en español para la interfaz
    df["stress_es"] = df["stress_level"].map(STRESS_LABELS_ES)
    df["addiction_es"] = df["addiction_level"].map(ADDICTION_LABELS_ES)
    df["gender_es"] = df["gender"].map(GENDER_LABELS_ES)
    df["impact_es"] = df["academic_work_impact"].map(IMPACT_LABELS_ES)

    cleaning_report: dict[str, object] = {
        "ruta_archivo": resolved_path.name,
        "filas_iniciales": filas_iniciales,
        "filas_finales": int(len(df)),
        "columnas_totales": int(len(EXPECTED_COLUMNS)),
        "nulos_addiction_colab": default_nulls_addiction,
        "nulos_restantes": int(df[list(EXPECTED_COLUMNS)].isna().sum().sum()),
        "duplicados_eliminados": duplicados_eliminados,
        "categorias_inesperadas": invalid_summary,
    }
    return df, cleaning_report


def filter_dataset(
    df: pd.DataFrame,
    age_range: tuple[int, int],
    genders: list[str],
    stress_levels: list[str],
    addiction_levels: list[str],
) -> pd.DataFrame:
    """Filtra el DataFrame según los controles interactivos del usuario."""
    min_age, max_age = age_range
    mask = (
        (df["age"] >= min_age)
        & (df["age"] <= max_age)
        & (df["gender"].isin(genders))
        & (df["stress_level"].isin(stress_levels))
        & (df["addiction_level"].isin(addiction_levels))
    )
    return df.loc[mask].copy()


# ==========================================================================
# src/analytics.py — Estadisticas descriptivas con NumPy y agregaciones pandas
# ==========================================================================

NUMERIC_VARIABLES_ES: dict[str, str] = {
    "daily_screen_time_hours": "Uso diario de pantalla (h)",
    "social_media_hours": "Redes sociales (h/día)",
    "gaming_hours": "Videojuegos (h/día)",
    "work_study_hours": "Estudio / trabajo (h/día)",
    "sleep_hours": "Horas de sueño (h/día)",
    "weekend_screen_time": "Pantalla en fin de semana (h)",
    "notifications_per_day": "Notificaciones por día",
    "app_opens_per_day": "Aperturas de apps por día",
    "age": "Edad (años)",
}

CORRELATION_COLUMNS_ES: dict[str, str] = {
    "daily_screen_time_hours": "Pantalla diaria",
    "social_media_hours": "Redes sociales",
    "gaming_hours": "Videojuegos",
    "work_study_hours": "Estudio/Trabajo",
    "sleep_hours": "Sueño",
    "notifications_per_day": "Notificaciones",
    "app_opens_per_day": "Aperturas app",
    "weekend_screen_time": "Fin de semana",
    "addicted_label": "Adicción (0/1)",
    "stress_score": "Estrés (0-2)",
    "impact_binary": "Impacto acad. (0/1)",
}


def compute_numpy_summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Calcula 8 estadísticas descriptivas básicas usando funciones de NumPy."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "conteo": 0,
            "media": 0.0,
            "mediana": 0.0,
            "desviacion": 0.0,
            "minimo": 0.0,
            "maximo": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    return {
        "conteo": int(arr.size),
        "media": round(float(np.mean(arr)), 2),
        "mediana": round(float(np.median(arr)), 2),
        "desviacion": round(float(np.std(arr)), 2),
        "minimo": round(float(np.min(arr)), 2),
        "maximo": round(float(np.max(arr)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
    }


def build_numpy_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla resumen en pandas recorriendo las variables clave con NumPy."""
    rows: list[dict[str, object]] = []
    for col, label in NUMERIC_VARIABLES_ES.items():
        stats = compute_numpy_summary(df[col])
        rows.append(
            {
                "Variable": label,
                "Registros": stats["conteo"],
                "Media (NumPy)": stats["media"],
                "Mediana": stats["mediana"],
                "Desv. Estándar": stats["desviacion"],
                "Mínimo": stats["minimo"],
                "P25": stats["p25"],
                "P75": stats["p75"],
                "Máximo": stats["maximo"],
            }
        )
    return pd.DataFrame(rows)


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Calcula los indicadores principales del subconjunto filtrado."""
    if df.empty:
        return {
            "total": 0,
            "promedio_pantalla": 0.0,
            "promedio_redes": 0.0,
            "promedio_sueno": 0.0,
            "pct_adiccion": 0.0,
            "pct_severa": 0.0,
            "pct_estres_alto": 0.0,
            "pct_impacto_academico": 0.0,
        }

    return {
        "total": int(len(df)),
        "promedio_pantalla": round(float(np.mean(df["daily_screen_time_hours"])), 2),
        "promedio_redes": round(float(np.mean(df["social_media_hours"])), 2),
        "promedio_sueno": round(float(np.mean(df["sleep_hours"])), 2),
        "pct_adiccion": round(float(np.mean(df["addicted_label"]) * 100.0), 1),
        "pct_severa": round(float(np.mean(df["addiction_level"] == "Severe") * 100.0), 1),
        "pct_estres_alto": round(float(np.mean(df["stress_level"] == "High") * 100.0), 1),
        "pct_impacto_academico": round(float(np.mean(df["impact_binary"]) * 100.0), 1),
    }


def compute_bracket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa métricas clave por tramo de tiempo diario en pantalla."""
    records: list[dict[str, object]] = []
    for bracket in SCREEN_BRACKET_ORDER:
        subset = df[df["screen_bracket"] == bracket]
        n = int(len(subset))
        if n == 0:
            records.append(
                {
                    "Tramo": bracket,
                    "Usuarios": 0,
                    "% Adicción (Mod/Sev)": 0.0,
                    "% Adicción Severa": 0.0,
                    "% Estrés Alto": 0.0,
                    "% Impacto Académico": 0.0,
                    "Sueño Promedio (h)": 0.0,
                }
            )
            continue

        records.append(
            {
                "Tramo": bracket,
                "Usuarios": n,
                "% Adicción (Mod/Sev)": round(float(np.mean(subset["addicted_label"]) * 100.0), 1),
                "% Adicción Severa": round(float(np.mean(subset["addiction_level"] == "Severe") * 100.0), 1),
                "% Estrés Alto": round(float(np.mean(subset["stress_level"] == "High") * 100.0), 1),
                "% Impacto Académico": round(float(np.mean(subset["impact_binary"]) * 100.0), 1),
                "Sueño Promedio (h)": round(float(np.mean(subset["sleep_hours"])), 2),
            }
        )
    return pd.DataFrame(records)


def compute_hourly_addiction_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la evolución hora a hora (3h a 12h) de adicción y severidad."""
    rows: list[dict[str, object]] = []
    for hour in range(3, 12):
        upper = hour + 1
        if upper == 12:
            subset = df[(df["daily_screen_time_hours"] >= hour) & (df["daily_screen_time_hours"] <= upper)]
        else:
            subset = df[(df["daily_screen_time_hours"] >= hour) & (df["daily_screen_time_hours"] < upper)]

        if subset.empty:
            continue

        rows.append(
            {
                "Rango": f"{hour}-{upper}h",
                "Hora_centro": hour + 0.5,
                "Usuarios": int(len(subset)),
                "% Adicción (Mod/Sev)": round(float(np.mean(subset["addicted_label"]) * 100.0), 1),
                "% Severa": round(float(np.mean(subset["addiction_level"] == "Severe") * 100.0), 1),
                "% Sin Adicción": round(float(np.mean(subset["addiction_level"] == "None") * 100.0), 1),
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson con np.corrcoef."""
    cols = list(CORRELATION_COLUMNS_ES.keys())
    labels = [CORRELATION_COLUMNS_ES[c] for c in cols]
    if len(df) < 2:
        return pd.DataFrame(np.zeros((len(cols), len(cols))), index=labels, columns=labels)

    matrix_data = df[cols].to_numpy(dtype=float)
    corr = np.corrcoef(matrix_data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return pd.DataFrame(np.round(corr, 2), index=labels, columns=labels)


def compute_cohort_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compara la cohorte joven (18-25) frente a adultos jóvenes (26-35)."""
    cohorts = [
        ("Jóvenes (18 - 25 años)", df[(df["age"] >= 18) & (df["age"] <= 25)]),
        ("Adultos jóvenes (26 - 35 años)", df[(df["age"] >= 26) & (df["age"] <= 35)]),
    ]
    rows: list[dict[str, object]] = []
    for label, sub in cohorts:
        kpis = compute_kpis(sub)
        rows.append(
            {
                "Cohorte": label,
                "Registros": kpis["total"],
                "Pantalla media (h)": kpis["promedio_pantalla"],
                "Redes media (h)": kpis["promedio_redes"],
                "Sueño medio (h)": kpis["promedio_sueno"],
                "% Adicción": kpis["pct_adiccion"],
                "% Estrés Alto": kpis["pct_estres_alto"],
                "% Impacto Académico": kpis["pct_impacto_academico"],
            }
        )
    return pd.DataFrame(rows)


def compute_addiction_means(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula promedios de uso según el nivel de adicción."""
    rows: list[dict[str, object]] = []
    for level in ADDICTION_ORDER:
        sub = df[df["addiction_level"] == level]
        if sub.empty:
            continue
        rows.append(
            {
                "Nivel": level,
                "Usuarios": int(len(sub)),
                "Pantalla (h)": round(float(np.mean(sub["daily_screen_time_hours"])), 2),
                "Redes (h)": round(float(np.mean(sub["social_media_hours"])), 2),
                "Videojuegos (h)": round(float(np.mean(sub["gaming_hours"])), 2),
                "Estudio (h)": round(float(np.mean(sub["work_study_hours"])), 2),
                "Sueño (h)": round(float(np.mean(sub["sleep_hours"])), 2),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# src/charts.py — Visualizaciones en Matplotlib
# ==========================================================================

PALETTE = {
    "primary": "#1D4ED8",
    "secondary": "#0EA5E9",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "success": "#16A34A",
    "neutral": "#64748B",
}

ADDICTION_COLORS: dict[str, str] = {
    "None": "#16A34A",
    "Mild": "#0EA5E9",
    "Moderate": "#F59E0B",
    "Severe": "#DC2626",
}


def _style_axes(ax: plt.Axes) -> None:
    """Aplica estilo limpio y consistente a los ejes de Matplotlib."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94A3B8")
    ax.set_axisbelow(True)


def plot_bracket_indicators(bracket_df: pd.DataFrame) -> Figure:
    """Gráfico de barras agrupadas: tramos de pantalla vs adicción, estrés e impacto."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _style_axes(ax)

    labels = bracket_df["Tramo"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    metrics = [
        ("% Adicción (Mod/Sev)", PALETTE["danger"], "Adicción (Mod/Sev)"),
        ("% Estrés Alto", PALETTE["warning"], "Estrés alto"),
        ("% Impacto Académico", PALETTE["primary"], "Impacto académico"),
    ]

    for idx, (col, color, legend_label) in enumerate(metrics):
        values = bracket_df[col].to_numpy(dtype=float)
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width=width, label=legend_label, color=color, alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1E293B",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Porcentaje de usuarios (%)")
    ax.set_title("Indicadores clave según tramo diario de pantalla", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_hourly_addiction_curve(hourly_df: pd.DataFrame) -> Figure:
    """Gráfico de líneas: evolución hora a hora de la adicción y severidad."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    _style_axes(ax)

    x = np.arange(len(hourly_df))
    ax.plot(
        x,
        hourly_df["% Adicción (Mod/Sev)"],
        marker="o",
        linewidth=2.4,
        color=PALETTE["danger"],
        label="% Adicción (Moderada o Severa)",
    )
    ax.plot(
        x,
        hourly_df["% Severa"],
        marker="s",
        linewidth=2.2,
        color=PALETTE["warning"],
        label="% Adicción Severa",
    )
    ax.plot(
        x,
        hourly_df["% Sin Adicción"],
        marker="^",
        linewidth=2.0,
        linestyle="--",
        color=PALETTE["success"],
        label="% Sin Adicción (None)",
    )

    # Líneas verticales en los saltos empíricos (6-7h y 8-9h)
    ranges = hourly_df["Rango"].tolist()
    if "6-7h" in ranges:
        idx_6 = ranges.index("6-7h")
        ax.axvline(idx_6, color="#64748B", linestyle=":", linewidth=1.3)
        ax.text(idx_6 + 0.08, 92, "Salto > 6h", fontsize=8.5, color="#475569")
    if "8-9h" in ranges:
        idx_8 = ranges.index("8-9h")
        ax.axvline(idx_8, color="#991B1B", linestyle=":", linewidth=1.3)
        ax.text(idx_8 + 0.08, 78, "100% adicción > 8h", fontsize=8.5, color="#991B1B")

    ax.set_xticks(x)
    ax.set_xticklabels(ranges, fontsize=9)
    ax.set_ylim(-3, 118)
    ax.set_xlabel("Rango de horas diarias de pantalla")
    ax.set_ylabel("Porcentaje dentro del rango (%)")
    ax.set_title("Puntos de quiebre: cómo escala la adicción por hora de uso", fontsize=12, fontweight="bold", pad=22)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=8.2)
    fig.tight_layout()
    return fig


def plot_scatter_by_addiction(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
) -> Figure:
    """Gráfico de dispersión entre uso diario de pantalla y variable seleccionada."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _style_axes(ax)
    ax.grid(True, linestyle="--", alpha=0.25, color="#94A3B8")

    # Muestreamos hasta 1,800 puntos si el conjunto es muy grande para mantener claridad visual
    plot_df = df.sample(n=1800, random_state=42) if len(df) > 1800 else df

    for level in ADDICTION_ORDER:
        sub = plot_df[plot_df["addiction_level"] == level]
        if sub.empty:
            continue
        ax.scatter(
            sub["daily_screen_time_hours"],
            sub[y_col],
            s=22,
            alpha=0.55,
            color=ADDICTION_COLORS[level],
            label=ADDICTION_LABELS_ES[level],
            edgecolors="none",
        )

    ax.axvline(6.0, color="#64748B", linestyle=":", linewidth=1.1)
    ax.axvline(8.0, color="#DC2626", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Uso diario de pantalla (horas)")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Relación entre pantalla diaria y {y_label.lower()} por nivel de adicción",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(title="Nivel de adicción", frameon=False, loc="upper left", ncol=4, fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_boxplot_by_addiction(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
) -> Figure:
    """Diagrama de cajas comparando una variable según el nivel de adicción."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    _style_axes(ax)

    groups: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []

    for level in ADDICTION_ORDER:
        values = df.loc[df["addiction_level"] == level, metric_col].to_numpy(dtype=float)
        if values.size > 0:
            groups.append(values)
            labels.append(ADDICTION_LABELS_ES[level])
            colors.append(ADDICTION_COLORS[level])

    if groups:
        try:
            bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5)
        except TypeError:
            bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("#1E293B")
        for median in bp["medians"]:
            median.set_color("#0F172A")
            median.set_linewidth(2.0)

    ax.set_ylabel(metric_label)
    ax.set_title(f"Distribución de '{metric_label}' por nivel de adicción", fontsize=11.5, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Figure:
    """Mapa de calor en Matplotlib para la matriz de correlaciones de NumPy."""
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    values = corr_df.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    labels = corr_df.columns.tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = values[i, j]
            text_color = "#FFFFFF" if abs(val) >= 0.55 else "#0F172A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación de Pearson (r)", fontsize=9)
    ax.set_title("Matriz de correlaciones (NumPy np.corrcoef)", fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


def plot_user_vs_cohort_comparison(
    user_profile: dict[str, float | int],
    reference_df: pd.DataFrame,
) -> Figure:
    """Barras horizontales comparando los hábitos del usuario frente a la cohorte."""
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    _style_axes(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#94A3B8")
    ax.grid(axis="y", visible=False)

    items = [
        ("daily_screen_time_hours", "Pantalla diaria (h)"),
        ("social_media_hours", "Redes sociales (h)"),
        ("gaming_hours", "Videojuegos (h)"),
        ("work_study_hours", "Estudio / Trabajo (h)"),
        ("sleep_hours", "Sueño (h)"),
    ]

    labels = [label for _, label in items]
    user_vals = [float(user_profile.get(col, 0.0)) for col, _ in items]
    cohort_vals = [
        float(np.mean(reference_df[col])) if not reference_df.empty else 0.0
        for col, _ in items
    ]

    y = np.arange(len(labels))
    height = 0.35

    ax.barh(y + height / 2, user_vals, height=height, color=PALETTE["primary"], label="Tu rutina")
    ax.barh(y - height / 2, cohort_vals, height=height, color=PALETTE["neutral"], alpha=0.75, label="Promedio cohorte")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Horas por día")
    ax.set_title("Tu perfil diario comparado contra la muestra seleccionada", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


# ==========================================================================
# src/guardian_engine.py — Simulador de habitos por reglas y persistencia JSON
# ==========================================================================

HISTORY_FILE = Path(__file__).resolve().parent / "data" / "historial_evaluaciones.json"

# Diccionario de umbrales: define alertas por hábito (máximo recomendado o mínimo sano)
MAX_LIMIT_RULES: dict[str, dict[str, object]] = {
    "daily_screen_time_hours": {
        "label": "Tiempo diario en pantalla",
        "warn": 6.0,
        "critical": 8.0,
        "unit": "h/día",
        "points_warn": 15,
        "points_critical": 35,
        "advice": "Reducir bloques de pantalla continua y programar pausas cada 50 minutos.",
    },
    "social_media_hours": {
        "label": "Redes sociales",
        "warn": 3.5,
        "critical": 5.0,
        "unit": "h/día",
        "points_warn": 12,
        "points_critical": 25,
        "advice": "Limitar el acceso a redes sociales durante franjas de estudio o clase.",
    },
    "gaming_hours": {
        "label": "Videojuegos",
        "warn": 2.0,
        "critical": 3.0,
        "unit": "h/día",
        "points_warn": 8,
        "points_critical": 15,
        "advice": "Reservar sesiones de juego para después de cumplir metas académicas.",
    },
    "notifications_per_day": {
        "label": "Notificaciones diarias",
        "warn": 120.0,
        "critical": 180.0,
        "unit": "notif/día",
        "points_warn": 6,
        "points_critical": 12,
        "advice": "Silenciar alertas no urgentes y agrupar notificaciones por horarios.",
    },
}


def evaluate_user_habits(
    user_profile: dict[str, float | int],
    reference_df: pd.DataFrame,
) -> dict[str, object]:
    """Evalúa los hábitos ingresados usando reglas condicionales y pares del dataset."""
    score = 0
    rule_results: list[dict[str, str]] = []
    recommendations: list[str] = []

    # 1. Evaluación mediante bucle sobre reglas de máximos (pantalla, redes, juegos, notif.)
    for key, rule in MAX_LIMIT_RULES.items():
        value = float(user_profile.get(key, 0.0))
        warn_lim = float(rule["warn"])
        crit_lim = float(rule["critical"])

        if value >= crit_lim:
            status = "Crítico"
            score += int(rule["points_critical"])
            recommendations.append(str(rule["advice"]))
        elif value >= warn_lim:
            status = "Precaución"
            score += int(rule["points_warn"])
            recommendations.append(str(rule["advice"]))
        else:
            status = "Óptimo"

        rule_results.append(
            {
                "Indicador": str(rule["label"]),
                "Tu valor": f"{value:.1f} {rule['unit']}" if key != "notifications_per_day" else f"{int(value)} {rule['unit']}",
                "Umbral aviso": f"{warn_lim:g} {rule['unit']}",
                "Estado": status,
            }
        )

    # 2. Regla de mínimo saludable para horas de sueño
    sleep_hours = float(user_profile.get("sleep_hours", 7.0))
    if sleep_hours < 6.0:
        sleep_status = "Crítico"
        score += 18
        recommendations.append(
            "Activar modo descanso nocturno 45 min antes de dormir para superar las 7h de sueño."
        )
    elif sleep_hours < 7.0:
        sleep_status = "Precaución"
        score += 8
        recommendations.append("Proteger la franja nocturna para evitar desplazar el sueño por pantalla.")
    else:
        sleep_status = "Óptimo"

    rule_results.append(
        {
            "Indicador": "Horas de sueño",
            "Tu valor": f"{sleep_hours:.1f} h/día",
            "Umbral aviso": ">= 7 h/día",
            "Estado": sleep_status,
        }
    )

    # 3. Balance ocio digital vs. estudio/trabajo
    study_hours = float(user_profile.get("work_study_hours", 3.0))
    leisure_hours = float(user_profile.get("social_media_hours", 0.0)) + float(
        user_profile.get("gaming_hours", 0.0)
    )
    if leisure_hours > study_hours * 1.5 and leisure_hours > 4.0:
        score += 10
        recommendations.append(
            "Tu tiempo combinado en redes y videojuegos supera ampliamente al de estudio/trabajo."
        )

    score = min(100, score)

    if score >= 60:
        risk_level = "Alerta Alta (Riesgo Digital)"
        device_message = (
            "Tu dispositivo detecta un patrón de sobrecarga. Si tuviera controles activos, "
            "restringiría temporalmente redes sociales y videojuegos hasta completar tus bloques de estudio y descanso."
        )
    elif score >= 30:
        risk_level = "Zona de Precaución"
        device_message = (
            "Vas en el límite donde los hábitos empiezan a escalar hacia dependencia digital. "
            "Es buen momento para poner límites antes de pasar las 6-8 horas diarias."
        )
    else:
        risk_level = "Uso Equilibrado"
        device_message = (
            "Tu relación con el celular muestra balance entre pantalla, estudio y descanso. "
            "Dispositivo inteligente, usuario inteligente."
        )

    if not recommendations:
        recommendations.append("Mantener tus hábitos actuales y revisar tu tiempo semanalmente.")

    # 4. Respaldo empírico calculado en el dataset real (cohorte similar +- 1.5h)
    user_screen = float(user_profile.get("daily_screen_time_hours", 5.0))
    if reference_df.empty:
        cohort_df = reference_df
        percentile = 0.0
    else:
        screen_arr = reference_df["daily_screen_time_hours"].to_numpy(dtype=float)
        percentile = round(float(np.mean(screen_arr <= user_screen) * 100.0), 1)
        cohort_df = reference_df[
            (reference_df["daily_screen_time_hours"] >= user_screen - 1.5)
            & (reference_df["daily_screen_time_hours"] <= user_screen + 1.5)
        ]
        if len(cohort_df) < 15:
            cohort_df = reference_df

    if cohort_df.empty:
        cohort_stats = {
            "cohorte_n": 0,
            "pct_adiccion": 0.0,
            "pct_severa": 0.0,
            "pct_estres_alto": 0.0,
            "pct_impacto_academico": 0.0,
        }
    else:
        cohort_stats = {
            "cohorte_n": int(len(cohort_df)),
            "pct_adiccion": round(float(np.mean(cohort_df["addicted_label"]) * 100.0), 1),
            "pct_severa": round(float(np.mean(cohort_df["addiction_level"] == "Severe") * 100.0), 1),
            "pct_estres_alto": round(float(np.mean(cohort_df["stress_level"] == "High") * 100.0), 1),
            "pct_impacto_academico": round(float(np.mean(cohort_df["impact_binary"]) * 100.0), 1),
        }

    return {
        "score": int(score),
        "risk_level": risk_level,
        "device_message": device_message,
        "percentile": percentile,
        "rule_results": rule_results,
        "recommendations": recommendations,
        "cohort_stats": cohort_stats,
    }


def load_evaluation_history(path: Path = HISTORY_FILE) -> list[dict[str, object]]:
    """Lee el historial de autoevaluaciones almacenado en formato JSON."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, OSError):
        return []


def save_evaluation_to_json(
    alias: str,
    user_profile: dict[str, float | int],
    evaluation: dict[str, object],
    path: Path = HISTORY_FILE,
) -> list[dict[str, object]]:
    """Guarda un registro resumido de la evaluación en un archivo JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_evaluation_history(path)

    cohort_stats = evaluation["cohort_stats"]
    clean_alias = alias.strip() or "Estudiante"

    entry: dict[str, object] = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "alias": clean_alias,
        "edad": int(user_profile.get("age", 20)),
        "pantalla_h": round(float(user_profile.get("daily_screen_time_hours", 0.0)), 1),
        "redes_h": round(float(user_profile.get("social_media_hours", 0.0)), 1),
        "juegos_h": round(float(user_profile.get("gaming_hours", 0.0)), 1),
        "estudio_h": round(float(user_profile.get("work_study_hours", 0.0)), 1),
        "sueno_h": round(float(user_profile.get("sleep_hours", 0.0)), 1),
        "puntaje_riesgo": int(evaluation["score"]),
        "diagnostico": str(evaluation["risk_level"]),
        "pct_adiccion_cohorte": float(cohort_stats["pct_adiccion"]),
    }

    history.append(entry)
    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    return history


def clear_evaluation_history(path: Path = HISTORY_FILE) -> None:
    """Limpia el archivo JSON de historial de evaluaciones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump([], file, ensure_ascii=False, indent=2)


# ==========================================================================
# app_2_p1zy.py — Aplicacion principal en Streamlit
# ==========================================================================

st.set_page_config(
    page_title="Uso Móvil y Desempeño Académico | Proyecto 1er Semestre",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_cleaned_dataset() -> tuple[pd.DataFrame, dict[str, object]]:
    """Carga y almacena en caché el dataset limpio para evitar lecturas repetidas."""
    return load_and_clean_data()


def _set_age_preset(min_age: int, max_age: int) -> None:
    """Actualiza el rango de edad en session_state al pulsar un botón rápido."""
    st.session_state["age_range_slider"] = (min_age, max_age)


def main() -> None:
    """Punto de entrada de la aplicación interactiva en Streamlit."""
    try:
        full_df, cleaning_report = get_cleaned_dataset()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Error al cargar el dataset: {exc}")
        st.stop()

    if "age_range_slider" not in st.session_state:
        st.session_state["age_range_slider"] = (18, 25)

    # --- Encabezado principal ---
    st.title("📱 Uso de Dispositivos Móviles, Estrés y Desempeño Académico")
    st.markdown(
        "**Pregunta guía:** *¿Cómo afecta el tiempo diario de uso de dispositivos móviles "
        "al nivel de estrés y al desempeño académico de los jóvenes?*  \n"
        "**Proyecto Integrador 1er Semestre** — Daniel C. & Santiago B. | "
        "*\"Dispositivos inteligentes para gente inteligente\"*"
    )

    # --- Barra lateral con filtros interactivos (R5) ---
    st.sidebar.header("🎛️ Filtros de Exploración")
    st.sidebar.caption("Por defecto se destaca la cohorte más joven (**18 a 25 años**). Usa el control deslizante para ampliar o ajustar el muestreo.")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    col_btn1.button(
        "Jóvenes (18–25)",
        on_click=_set_age_preset,
        args=(18, 25),
        use_container_width=True,
    )
    col_btn2.button(
        "Todos (18–35)",
        on_click=_set_age_preset,
        args=(18, 35),
        use_container_width=True,
    )

    age_range = st.sidebar.slider(
        "Rango de edad (años)",
        min_value=int(full_df["age"].min()),
        max_value=int(full_df["age"].max()),
        key="age_range_slider",
    )

    gender_options = list(GENDER_LABELS_ES.keys())
    selected_genders = st.sidebar.multiselect(
        "Género",
        options=gender_options,
        default=gender_options,
        format_func=lambda g: GENDER_LABELS_ES.get(g, g),
    )

    selected_stress = st.sidebar.multiselect(
        "Nivel de estrés",
        options=STRESS_ORDER,
        default=STRESS_ORDER,
        format_func=lambda s: STRESS_LABELS_ES.get(s, s),
    )

    selected_addiction = st.sidebar.multiselect(
        "Nivel de adicción",
        options=ADDICTION_ORDER,
        default=ADDICTION_ORDER,
        format_func=lambda a: ADDICTION_LABELS_ES.get(a, a),
    )

    filtered_df = filter_dataset(
        full_df,
        age_range=age_range,
        genders=selected_genders,
        stress_levels=selected_stress,
        addiction_levels=selected_addiction,
    )

    st.sidebar.divider()
    pct_sample = (len(filtered_df) / len(full_df)) * 100 if len(full_df) > 0 else 0.0
    st.sidebar.metric(
        "Muestra activa",
        f"{len(filtered_df):,} / {len(full_df):,}",
        f"{pct_sample:.1f}% del dataset",
    )

    if filtered_df.empty:
        st.warning("No hay registros que coincidan con los filtros seleccionados. Ajusta la barra lateral.")
        st.stop()

    kpis = compute_kpis(filtered_df)

    # --- Pestañas principales ---
    tab_story, tab_numpy, tab_simulator, tab_method = st.tabs(
        [
            "📖 Historia y Hallazgos",
            "🧮 Laboratorio NumPy & EDA",
            "🛡️ Simulador de Hábitos",
            "🧭 Metodología y Visión",
        ]
    )

    # =========================================================================
    # PESTAÑA 1: HISTORIA Y HALLAZGOS
    # =========================================================================
    with tab_story:
        if age_range == (18, 25):
            st.info(
                f"Estudio enfocado en la **cohorte joven destacada (18–25 años)**: "
                f"**{len(filtered_df):,}** registros seleccionados."
            )
        else:
            st.info(
                f"Estudio sobre el rango de edad **{age_range[0]}–{age_range[1]} años**: "
                f"**{len(filtered_df):,}** registros seleccionados."
            )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Registros", f"{kpis['total']:,}")
        m2.metric("Pantalla diaria (media)", f"{kpis['promedio_pantalla']:.2f} h")
        m3.metric("Adicción (Mod/Sev)", f"{kpis['pct_adiccion']:.1f}%")
        m4.metric("Estrés alto", f"{kpis['pct_estres_alto']:.1f}%")
        m5.metric("Impacto académico", f"{kpis['pct_impacto_academico']:.1f}%")

        st.subheader("1. ¿Qué pasa conforme aumentan las horas diarias de pantalla?")
        bracket_df = compute_bracket_summary(filtered_df)
        hourly_df = compute_hourly_addiction_curve(filtered_df)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_bracket = plot_bracket_indicators(bracket_df)
            st.pyplot(fig_bracket)
            plt.close(fig_bracket)
        with col_g2:
            if not hourly_df.empty:
                fig_hourly = plot_hourly_addiction_curve(hourly_df)
                st.pyplot(fig_hourly)
                plt.close(fig_hourly)

        with st.expander("Ver tabla numérica por tramos de pantalla"):
            st.dataframe(bracket_df, use_container_width=True, hide_index=True)

        st.subheader("2. Exploración interactiva por hábitos y nivel de adicción")
        col_sel1, col_sel2 = st.columns(2)
        scatter_choices = {
            "weekend_screen_time": "Pantalla en fin de semana (horas)",
            "sleep_hours": "Horas de sueño",
            "work_study_hours": "Horas de estudio / trabajo",
            "notifications_per_day": "Notificaciones por día",
        }
        boxplot_choices = {
            "social_media_hours": "Horas en redes sociales",
            "daily_screen_time_hours": "Uso diario de pantalla (h)",
            "gaming_hours": "Horas en videojuegos",
            "sleep_hours": "Horas de sueño",
            "work_study_hours": "Horas de estudio / trabajo",
        }

        chosen_scatter_y = col_sel1.selectbox(
            "Variable eje Y (dispersión):",
            options=list(scatter_choices.keys()),
            format_func=lambda k: scatter_choices[k],
        )
        chosen_box_metric = col_sel2.selectbox(
            "Variable a comparar (cajas):",
            options=list(boxplot_choices.keys()),
            format_func=lambda k: boxplot_choices[k],
        )

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            fig_scatter = plot_scatter_by_addiction(
                filtered_df, chosen_scatter_y, scatter_choices[chosen_scatter_y]
            )
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
        with col_g4:
            fig_box = plot_boxplot_by_addiction(
                filtered_df, chosen_box_metric, boxplot_choices[chosen_box_metric]
            )
            st.pyplot(fig_box)
            plt.close(fig_box)

        st.subheader("3. Respuesta basada en los datos a la Pregunta Guía")
        low_row = bracket_df.loc[bracket_df["Tramo"] == "Moderado (< 5h)"].iloc[0]
        high_row = bracket_df.loc[bracket_df["Tramo"] == "Crítico (> 8h)"].iloc[0]

        if low_row["Usuarios"] > 0 and high_row["Usuarios"] > 0:
            st.success(
                f"**Evidencia empírica del dataset:**  \n"
                f"- **Relación directa con la dependencia digital:** Pasar de menos de 5 horas diarias a más de 8 horas "
                f"dispara el porcentaje de adicción (Moderada/Severa) de **{low_row['% Adicción (Mod/Sev)']:.1f}%** a "
                f"**{high_row['% Adicción (Mod/Sev)']:.1f}%**, con dos puntos de quiebre claros a partir de las **6 horas** y las **8 horas** diarias "
                f"(impulsado principalmente por redes sociales y uso en fin de semana).  \n"
                f"- **Hallazgo crítico sobre estrés e impacto académico en este dataset:** El impacto académico reportado "
                f"(**{low_row['% Impacto Académico']:.1f}%** vs **{high_row['% Impacto Académico']:.1f}%**) y el estrés alto "
                f"(**{low_row['% Estrés Alto']:.1f}%** vs **{high_row['% Estrés Alto']:.1f}%**) permanecen casi constantes entre tramos. "
                f"Esto revela una limitación importante del conjunto de datos público (posible generación sintética independiente en esas dos columnas), "
                f"aspecto clave para reflexionar antes de asumir causalidad directa."
            )
        else:
            st.info(
                "Amplía los filtros de la barra lateral para incluir registros en los tramos `< 5h` y `> 8h` "
                "y visualizar la comparativa completa entre extremos de uso."
            )

        st.subheader("4. Comparativa: Jóvenes (18–25) vs. Adultos jóvenes (26–35)")
        cohort_df = compute_cohort_comparison(full_df)
        st.dataframe(cohort_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # PESTAÑA 2: LABORATORIO NUMPY & EDA
    # =========================================================================
    with tab_numpy:
        st.subheader("Estadísticas descriptivas con NumPy (`np.mean`, `np.median`, `np.std`, `np.min`, `np.max`, `np.percentile`)")
        selected_var = st.selectbox(
            "Selecciona una variable para inspeccionar en detalle:",
            options=list(NUMERIC_VARIABLES_ES.keys()),
            format_func=lambda k: NUMERIC_VARIABLES_ES[k],
        )

        stats = compute_numpy_summary(filtered_df[selected_var])
        s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
        s1.metric("Media", f"{stats['media']}")
        s2.metric("Mediana", f"{stats['mediana']}")
        s3.metric("Desv. Est.", f"{stats['desviacion']}")
        s4.metric("Mínimo", f"{stats['minimo']}")
        s5.metric("P25", f"{stats['p25']}")
        s6.metric("P75", f"{stats['p75']}")
        s7.metric("Máximo", f"{stats['maximo']}")

        st.markdown("**Resumen general de todas las variables numéricas (muestra filtrada):**")
        numpy_table = build_numpy_summary_table(filtered_df)
        st.dataframe(numpy_table, use_container_width=True, hide_index=True)

        st.subheader("Matriz de correlaciones (`np.corrcoef`) y promedios por nivel de adicción")
        col_hm, col_means = st.columns([1.25, 1.0])
        with col_hm:
            corr_df = compute_correlation_matrix(filtered_df)
            fig_hm = plot_correlation_heatmap(corr_df)
            st.pyplot(fig_hm)
            plt.close(fig_hm)
        with col_means:
            means_df = compute_addiction_means(filtered_df)
            st.markdown("**Promedios de hábito por nivel de adicción:**")
            st.dataframe(means_df, use_container_width=True, hide_index=True)
            st.caption(
                "Observa cómo la media de horas de pantalla y redes sociales crece claramente "
                "desde 'None' hacia 'Moderate' y 'Severe', mientras que el resto de variables "
                "muestran promedios casi planos en este dataset."
            )

        st.subheader("Exploración directa del DataFrame (`head`) y descarga CSV")
        n_rows = st.slider("Cantidad de filas a visualizar", min_value=5, max_value=50, value=10, step=5)
        st.dataframe(filtered_df.head(n_rows), use_container_width=True, hide_index=True)

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar muestra filtrada en CSV",
            data=csv_bytes,
            file_name="muestra_uso_movil_filtrada.csv",
            mime="text/csv",
        )

    # =========================================================================
    # PESTAÑA 3: SIMULADOR DE HÁBITOS (ALCANCE 1ER SEMESTRE)
    # =========================================================================
    with tab_simulator:
        st.subheader("🛡️ Simulador de Hábitos: *\"Dispositivos inteligentes para gente inteligente\"*")
        st.caption(
            "Prototipo educativo basado en reglas condicionales (`if/elif/else`), diccionarios de umbrales "
            "y comparación estadística con los datos reales. No accede ni bloquea aplicaciones reales del teléfono; "
            "proyecta tu nivel de riesgo y muestra qué consecuencias/patrones presentan usuarios con hábitos similares en el dataset."
        )

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            sim_alias = st.text_input("Nombre o alias (para guardar en JSON)", value="Estudiante")
            sim_age = st.number_input("Edad", min_value=15, max_value=60, value=20, step=1)
            sim_screen = st.slider("Uso diario de pantalla (h)", 1.0, 14.0, 7.0, 0.5)
        with col_in2:
            sim_social = st.slider("Redes sociales (h/día)", 0.0, 8.0, 3.5, 0.5)
            sim_gaming = st.slider("Videojuegos (h/día)", 0.0, 6.0, 1.5, 0.5)
            sim_study = st.slider("Estudio / trabajo productivo (h/día)", 0.5, 10.0, 3.0, 0.5)
        with col_in3:
            sim_sleep = st.slider("Horas de sueño (h/día)", 3.5, 10.0, 6.5, 0.5)
            sim_notif = st.slider("Notificaciones recibidas al día", 10, 300, 130, 10)

        user_profile = {
            "age": sim_age,
            "daily_screen_time_hours": sim_screen,
            "social_media_hours": sim_social,
            "gaming_hours": sim_gaming,
            "work_study_hours": sim_study,
            "sleep_hours": sim_sleep,
            "notifications_per_day": sim_notif,
        }

        evaluation = evaluate_user_habits(user_profile, filtered_df)
        cohort_stats = evaluation["cohort_stats"]

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Puntaje de riesgo (reglas)", f"{evaluation['score']} / 100")
        r2.metric("Diagnóstico", str(evaluation["risk_level"]))
        r3.metric("Percentil de uso", f"Superas al {evaluation['percentile']:.1f}%")
        r4.metric("Adicción en pares (±1.5h)", f"{cohort_stats['pct_adiccion']:.1f}%")

        if evaluation["score"] >= 60:
            st.error(f"**Mensaje del dispositivo:** {evaluation['device_message']}")
        elif evaluation["score"] >= 30:
            st.warning(f"**Mensaje del dispositivo:** {evaluation['device_message']}")
        else:
            st.success(f"**Mensaje del dispositivo:** {evaluation['device_message']}")

        col_res1, col_res2 = st.columns([1.05, 1.0])
        with col_res1:
            st.markdown("**Evaluación regla por regla:**")
            st.dataframe(pd.DataFrame(evaluation["rule_results"]), use_container_width=True, hide_index=True)
            st.markdown(
                f"**Lo que muestran {cohort_stats['cohorte_n']:,} registros similares (±1.5h en el dataset):** "
                f"**{cohort_stats['pct_adiccion']:.1f}%** presenta adicción moderada/severa "
                f"(**{cohort_stats['pct_severa']:.1f}%** severa), **{cohort_stats['pct_estres_alto']:.1f}%** reporta estrés alto "
                f"y **{cohort_stats['pct_impacto_academico']:.1f}%** reporta impacto académico."
            )
        with col_res2:
            fig_user = plot_user_vs_cohort_comparison(user_profile, filtered_df)
            st.pyplot(fig_user)
            plt.close(fig_user)

        st.markdown("**Acciones recomendadas (simulación preventiva):**")
        for rec in evaluation["recommendations"]:
            st.markdown(f"- {rec}")

        st.divider()
        st.markdown("**Registro persistente en JSON (`data/historial_evaluaciones.json`):**")
        col_j1, col_j2 = st.columns([1, 1])
        if col_j1.button("💾 Guardar esta evaluación en JSON", use_container_width=True):
            save_evaluation_to_json(sim_alias, user_profile, evaluation)
            st.toast("Evaluación guardada correctamente en JSON.")
        if col_j2.button("🗑️ Limpiar historial JSON", use_container_width=True):
            clear_evaluation_history()
            st.toast("Historial JSON limpiado.")

        history_records = load_evaluation_history()
        if history_records:
            st.dataframe(pd.DataFrame(history_records), use_container_width=True, hide_index=True)
        else:
            st.caption("Aún no hay evaluaciones guardadas en el historial JSON.")

    # =========================================================================
    # PESTAÑA 4: METODOLOGÍA, ÉTICA Y VISIÓN FUTURA
    # =========================================================================
    with tab_method:
        st.subheader("1. Auditoría y limpieza del dataset (R1 - R3)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas iniciales", f"{cleaning_report['filas_iniciales']:,}")
        c2.metric("Filas limpias", f"{cleaning_report['filas_finales']:,}")
        c3.metric("Nulos detectados en Colab", f"{cleaning_report['nulos_addiction_colab']:,}")
        c4.metric("Nulos tras limpieza", f"{cleaning_report['nulos_restantes']}")

        st.markdown(
            """
            - **Hallazgo clave en la limpieza:** En Google Colab, `pandas.read_csv()` interpretaba la cadena literal `"None"`
              de la columna `addiction_level` como `NaN` (`819` registros). En esta aplicación se preserva explícitamente `"None"`
              como la categoría válida **"Sin adicción"**, evitando perder el 10.9% de la muestra.
            - **Colecciones nativas y modularidad:** El procesamiento se divide en funciones reutilizables dentro de `src/`
              utilizando diccionarios de mapeo y reglas, listas de resultados, tuplas de columnas esperadas y conjuntos (`set`) para validar categorías.
            """
        )

        st.subheader("2. Reflexión crítica sobre los datos y la IA")
        st.markdown(
            """
            - **Valor y límites del dataset público:** Este conjunto de 7,500 registros permite identificar umbrales claros entre
              horas de pantalla, uso de redes sociales y nivel de adicción (`r = +0.58`). Sin embargo, variables como `stress_level`
              y `academic_work_impact` muestran distribuciones prácticamente uniformes e independientes del tiempo de uso (`r ≈ 0.00`).
              Esto enseña una lección fundamental de Ingeniería de Datos: **antes de entrenar modelos o tomar decisiones reales,
              siempre debemos auditar si los datos reflejan fenómenos humanos reales o reglas sintéticas.**
            - **Ética y privacidad:** Si en el futuro esta herramienta midiera el uso real del celular de estudiantes, debería operar
              con consentimiento informado, anonimización estricta y procesamiento local para no vigilar ni estigmatizar al usuario.
            """
        )

        st.subheader("3. ¿Cómo puede crecer este proyecto más allá del 1er semestre?")
        st.markdown(
            """
            1. **Etapa 2 (Datos propios y Backend):** Recolectar encuestas anónimas reales en la universidad o exportaciones voluntarias de
               *Bienestar Digital* y conectar la lógica a una API (`FastAPI`) con base de datos (`SQLite`/`PostgreSQL`).
            2. **Etapa 3 (Prototipo real en navegador o Android):** Llevar el motor de reglas a una extensión de navegador o a una app
               Android usando `UsageStatsManager` para medir uso real por aplicación y ofrecer pausas conscientes contextuales en horarios de estudio.
            """
        )


if __name__ == "__main__":
    main()
