"""
Dashboard Ejecutivo — Predicción de Deserción Estudiantil
Persona 4: Interpretación y Visualización
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deserción Estudiantil — Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261"]
CLUSTER_NAMES = {
    0: "Apoyo familiar sólido",
    1: "Rezago académico",
    2: "Rendimiento académico",
    3: "Progreso avanzado",
}
CLUSTER_COLORS = {
    "Apoyo familiar sólido":  "#F4A261",
    "Rezago académico":       "#E84855",
    "Rendimiento académico":  "#3BB273",
    "Progreso avanzado":      "#2E86AB",
}

# ─────────────────────────────────────────────
# ESTILO CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(46,134,171,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    color: white;
    margin-bottom: 8px;
}
.metric-card .value { font-size: 2.2rem; font-weight: 700; color: #2E86AB; }
.metric-card .label { font-size: 0.85rem; color: #adb5bd; margin-top: 4px; }
.metric-card .delta { font-size: 0.8rem; color: #3BB273; margin-top: 2px; }

.risk-high   { background: linear-gradient(135deg, #4a0010, #6b0020); border-left: 4px solid #E84855; border-radius: 8px; padding: 16px; margin: 6px 0; color: white; }
.risk-medium { background: linear-gradient(135deg, #4a3000, #6b4500); border-left: 4px solid #F4A261; border-radius: 8px; padding: 16px; margin: 6px 0; color: white; }
.risk-low    { background: linear-gradient(135deg, #003020, #004030); border-left: 4px solid #3BB273; border-radius: 8px; padding: 16px; margin: 6px 0; color: white; }

.section-title {
    font-size: 1.5rem; font-weight: 700;
    color: #2E86AB; margin-bottom: 4px;
    border-bottom: 2px solid rgba(46,134,171,0.2);
    padding-bottom: 8px;
}

[data-testid="stSidebar"] { background: #0f0f1a; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA Y CACHÉ DE DATOS / MODELOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "..", "data", "processed", "clean_dataset.csv")
    if not os.path.exists(path):
        path = "data/processed/clean_dataset.csv"
    df = pd.read_csv(path)
    return df


@st.cache_resource
def build_models(df):
    X = df.drop(columns=["Target_binary"])
    y = df["Target_binary"]

    # KMeans
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    df_c = df.copy()
    df_c["Cluster"] = clusters
    df_c["Cluster_Name"] = df_c["Cluster"].map(CLUSTER_NAMES)
    df_c["PC1"] = components[:, 0]
    df_c["PC2"] = components[:, 1]

    # Neural Net
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_nn = StandardScaler()
    X_train_sc = scaler_nn.fit_transform(X_train)
    X_test_sc = scaler_nn.transform(X_test)

    model = Sequential([
        Input(shape=(X_train_sc.shape[1],)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_sc, y_train,
        validation_split=0.2, epochs=50, batch_size=32,
        callbacks=[early_stop], verbose=0,
    )

    y_pred_prob = model.predict(X_test_sc, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Continúa", "Deserción"], output_dict=True)

    return {
        "df_clusters": df_c,
        "kmeans": kmeans,
        "scaler_km": scaler_km,
        "scaler_nn": scaler_nn,
        "model": model,
        "history": history.history,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
        "auc": auc,
        "cm": cm,
        "report": report,
        "X_test": X_test,
        "feature_names": X.columns.tolist(),
        "pca_var": pca.explained_variance_ratio_,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Dashboard\n**Deserción Estudiantil**")
    st.markdown("---")
    page = st.radio(
        "Navegar a:",
        ["📊 Resumen Ejecutivo",
         "🔍 Exploración de Datos",
         "🤖 Resultados del Modelo",
         "👥 Perfiles de Estudiantes",
         "🔮 Predicción Individual",
         "💡 Recomendaciones"],
    )
    st.markdown("---")
    st.markdown("**Proyecto:** ML Consulting  \n**Fase:** Interpretación y Visualización  \n**Persona 4**")

# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
with st.spinner("Cargando datos y entrenando modelos…"):
    df = load_data()
    models = build_models(df)
    df_c = models["df_clusters"]


# ══════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════
if page == "📊 Resumen Ejecutivo":
    st.markdown('<p class="section-title">📊 Resumen Ejecutivo</p>', unsafe_allow_html=True)
    st.caption("Predicción de Deserción Estudiantil — Hallazgos clave del proyecto")

    # KPIs
    total = len(df)
    dropout_pct = df["Target_binary"].mean() * 100
    high_risk = (df_c["Cluster_Name"] == "Rezago académico").sum()
    auc = models["auc"]
    accuracy = models["report"]["accuracy"] * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, str(total), "Estudiantes", "en el dataset"),
        (c2, f"{dropout_pct:.1f}%", "Tasa de Deserción", "promedio institucional"),
        (c3, str(high_risk), "En Alto Riesgo", "perfil rezago académico"),
        (c4, f"{accuracy:.1f}%", "Accuracy Modelo", "en datos de prueba"),
        (c5, f"{auc:.3f}", "AUC-ROC", "excelente > 0.9"),
    ]
    for col, val, label, delta in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
                <div class="delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribución de la Variable Objetivo")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["Target_binary"].value_counts()
        labels = ["Continúa (67.9%)", "Deserta (32.1%)"]
        ax.pie(counts.values, labels=labels, colors=[PALETTE[2], PALETTE[1]],
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
        ax.set_title("Estado Académico", fontweight="bold")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Tasa de Deserción por Perfil")
        cluster_drop = df_c.groupby("Cluster_Name")["Target_binary"].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = [CLUSTER_COLORS[n] for n in cluster_drop.index]
        bars = ax.barh(cluster_drop.index, cluster_drop.values, color=colors)
        ax.set_xlabel("Tasa de Deserción (%)")
        for bar, val in zip(bars, cluster_drop.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
        avg = dropout_pct
        ax.axvline(avg, color="gray", linestyle="--", alpha=0.7, label=f"Promedio ({avg:.1f}%)")
        ax.legend(fontsize=8)
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### 🏆 Hallazgos Principales")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown('<div class="risk-high"><strong>🔴 Mayor riesgo</strong><br>El perfil <em>Rezago académico</em> concentra 851 estudiantes con 82.7% de deserción. Es la prioridad de intervención número uno.</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="risk-medium"><strong>📚 Factor clave</strong><br>La calificación del 2do semestre (corr. 0.572) es el predictor más fuerte. El rendimiento en semestre 1 ya anticipa el riesgo.</div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="risk-low"><strong>💰 Factor protector</strong><br>Tener la colegiatura al corriente (corr. 0.429) reduce significativamente el riesgo. Los becados desertan menos.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PÁGINA 2 — EXPLORACIÓN DE DATOS
# ══════════════════════════════════════════════
elif page == "🔍 Exploración de Datos":
    st.markdown('<p class="section-title">🔍 Exploración de Datos</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Filtros**")
        show_dropout = st.multiselect(
            "Estado académico",
            options=["Continúa", "Deserta"],
            default=["Continúa", "Deserta"]
        )
        age_range = st.slider("Edad al inscribirse", int(df["Age at enrollment"].min()),
                               int(df["Age at enrollment"].max()), (17, 60))

    mask = pd.Series([True] * len(df))
    if "Continúa" not in show_dropout:
        mask &= df["Target_binary"] != 0
    if "Deserta" not in show_dropout:
        mask &= df["Target_binary"] != 1
    mask &= (df["Age at enrollment"] >= age_range[0]) & (df["Age at enrollment"] <= age_range[1])
    df_filtered = df[mask]

    with col2:
        st.metric("Estudiantes filtrados", len(df_filtered),
                  delta=f"{len(df_filtered)-len(df)} vs total")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribución de Edad")
        fig, ax = plt.subplots(figsize=(6, 4))
        for val, label, color in [(0, "Continúa", PALETTE[0]), (1, "Deserta", PALETTE[1])]:
            subset = df_filtered[df_filtered["Target_binary"] == val]
            ax.hist(subset["Age at enrollment"], bins=25, alpha=0.65, color=color, label=label)
        ax.set_xlabel("Edad"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.set_title("Edad al Inscribirse por Estado Académico", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown("#### Calificación 1er Semestre vs Deserción")
        fig, ax = plt.subplots(figsize=(6, 4))
        for val, label, color in [(0, "Continúa", PALETTE[0]), (1, "Deserta", PALETTE[1])]:
            subset = df_filtered[df_filtered["Target_binary"] == val]
            ax.hist(subset["Curricular units 1st sem (grade)"].replace(0, np.nan).dropna(),
                    bins=25, alpha=0.65, color=color, label=label)
        ax.set_xlabel("Calificación"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.set_title("Distribución de Calificaciones (Sem 1)", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("#### Top 10 Variables con Mayor Correlación con la Deserción")
    corr = df_filtered.corr()["Target_binary"].drop("Target_binary").abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE[1] if v > 0.4 else PALETTE[0] if v > 0.2 else "#B0BEC5" for v in corr.values]
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.set_xlabel("Correlación absoluta"); ax.set_title("Importancia de Variables", fontweight="bold")
    fig.patch.set_alpha(0); plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    with st.expander("Ver datos filtrados"):
        st.dataframe(df_filtered.head(200), use_container_width=True)


# ══════════════════════════════════════════════
# PÁGINA 3 — RESULTADOS DEL MODELO
# ══════════════════════════════════════════════
elif page == "🤖 Resultados del Modelo":
    st.markdown('<p class="section-title">🤖 Resultados del Modelo — Red Neuronal</p>', unsafe_allow_html=True)

    rep = models["report"]
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, f"{rep['accuracy']*100:.1f}%", "Accuracy"),
        (c2, f"{models['auc']:.3f}", "AUC-ROC"),
        (c3, f"{rep['Deserción']['recall']*100:.1f}%", "Recall (Deserción)"),
        (c4, f"{rep['Deserción']['precision']*100:.1f}%", "Precisión (Deserción)"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Curvas de Entrenamiento")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        h = models["history"]
        ax.plot(h["accuracy"], color=PALETTE[0], linewidth=2, label="Train")
        ax.plot(h["val_accuracy"], color=PALETTE[1], linewidth=2, linestyle="--", label="Validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Accuracy")
        ax.set_ylim([0.6, 1.0]); ax.legend()
        ax.set_title("Accuracy por Época", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("#### Curva ROC")
        fpr, tpr, _ = roc_curve(models["y_test"], models["y_pred_prob"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(fpr, tpr, color=PALETTE[0], linewidth=2, label=f"AUC={models['auc']:.3f}")
        ax.plot([0,1],[0,1], "k--", alpha=0.4)
        ax.set_xlabel("Falsos Positivos"); ax.set_ylabel("Verdaderos Positivos")
        ax.legend(); ax.set_title("Curva ROC", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col3:
        st.markdown("#### Matriz de Confusión")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        cm = models["cm"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Continúa", "Deserta"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Matriz de Confusión", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("#### Interpretación de Métricas en Lenguaje de Negocio")
    st.markdown("""
| Métrica | Valor | ¿Qué significa para la institución? |
|---------|-------|--------------------------------------|
| **Accuracy** | 87% | De cada 100 estudiantes, el modelo clasifica correctamente a 87 |
| **Recall (Deserción)** | 72% | Detecta 7 de cada 10 estudiantes que van a desertar |
| **Precisión (Deserción)** | 86% | Cuando dice "deserta", acierta el 86% de las veces |
| **AUC-ROC** | 0.929 | Excelente capacidad discriminatoria — 1.0 sería perfecto |
| **Falsos negativos** | ~79 | Estudiantes en riesgo NO detectados — área de mejora clave |
    """)


# ══════════════════════════════════════════════
# PÁGINA 4 — PERFILES
# ══════════════════════════════════════════════
elif page == "👥 Perfiles de Estudiantes":
    st.markdown('<p class="section-title">👥 Perfiles de Estudiantes (K-Means)</p>', unsafe_allow_html=True)
    st.caption("El algoritmo K-Means identificó 4 perfiles diferenciados con Silhouette Score = 0.217")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Mapa de Perfiles (PCA 2D)")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, color in CLUSTER_COLORS.items():
            mask = df_c["Cluster_Name"] == name
            ax.scatter(df_c.loc[mask, "PC1"], df_c.loc[mask, "PC2"],
                       c=color, alpha=0.4, s=15, label=name)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Visualización PCA", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("#### Resumen por Perfil")
        summary = df_c.groupby("Cluster_Name").agg(
            Estudiantes=("Target_binary", "count"),
            Desercion_pct=("Target_binary", lambda x: f"{x.mean()*100:.1f}%"),
            Edad_prom=("Age at enrollment", lambda x: f"{x.mean():.1f}"),
            Calif_sem1=("Curricular units 1st sem (grade)", lambda x: f"{x.mean():.2f}"),
            Calif_sem2=("Curricular units 2nd sem (grade)", lambda x: f"{x.mean():.2f}"),
            Pct_becados=("Scholarship holder", lambda x: f"{x.mean()*100:.1f}%"),
            Pct_deudores=("Debtor", lambda x: f"{x.mean()*100:.1f}%"),
        ).rename(columns={
            "Desercion_pct": "% Deserción",
            "Edad_prom": "Edad Prom",
            "Calif_sem1": "Calif. Sem1",
            "Calif_sem2": "Calif. Sem2",
            "Pct_becados": "% Becados",
            "Pct_deudores": "% Deudores",
        })
        st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    selected_cluster = st.selectbox("Selecciona un perfil para ver detalles:", list(CLUSTER_NAMES.values()))
    df_sel = df_c[df_c["Cluster_Name"] == selected_cluster]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudiantes", len(df_sel))
    c2.metric("Tasa Deserción", f"{df_sel['Target_binary'].mean()*100:.1f}%")
    c3.metric("Edad Promedio", f"{df_sel['Age at enrollment'].mean():.1f} años")
    c4.metric("Calif. Prom Sem1", f"{df_sel['Curricular units 1st sem (grade)'].mean():.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Distribución de calificaciones**")
        fig, ax = plt.subplots(figsize=(5, 3))
        vals = df_sel["Curricular units 1st sem (grade)"].replace(0, np.nan).dropna()
        ax.hist(vals, bins=20, color=CLUSTER_COLORS[selected_cluster], alpha=0.8)
        ax.set_xlabel("Calificación"); ax.set_ylabel("Frecuencia")
        ax.set_title("Calificaciones Semestre 1", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
    with col_b:
        st.markdown("**Variables binarias del perfil**")
        bin_vars = ["Debtor", "Scholarship holder", "Tuition fees up to date", "Gender", "Displaced"]
        means = df_sel[bin_vars].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(bin_vars, means.values, color=CLUSTER_COLORS[selected_cluster], alpha=0.8)
        ax.set_xlabel("% de estudiantes (valor=1)")
        ax.set_title("Variables Binarias", fontweight="bold")
        fig.patch.set_alpha(0); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════
# PÁGINA 5 — PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════
elif page == "🔮 Predicción Individual":
    st.markdown('<p class="section-title">🔮 Predicción Individual de Riesgo</p>', unsafe_allow_html=True)
    st.caption("Ingresa los datos de un estudiante para obtener su probabilidad de deserción")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Datos Académicos**")
        grade_1 = st.slider("Calificación Semestre 1", 0.0, 20.0, 12.0, 0.1)
        approved_1 = st.slider("Materias Aprobadas Sem 1", 0, 10, 5)
        enrolled_1 = st.slider("Materias Inscritas Sem 1", 0, 10, 6)
        grade_2 = st.slider("Calificación Semestre 2", 0.0, 20.0, 12.0, 0.1)
        approved_2 = st.slider("Materias Aprobadas Sem 2", 0, 10, 5)
        enrolled_2 = st.slider("Materias Inscritas Sem 2", 0, 10, 6)

    with col2:
        st.markdown("**👤 Datos Personales**")
        age = st.slider("Edad al inscribirse", 17, 70, 20)
        gender = st.selectbox("Género", ["Masculino (1)", "Femenino (0)"])
        gender_val = 1 if "Masculino" in gender else 0
        debtor = st.selectbox("¿Es deudor?", ["No (0)", "Sí (1)"])
        debtor_val = 1 if "Sí" in debtor else 0
        tuition = st.selectbox("¿Colegiatura al corriente?", ["Sí (1)", "No (0)"])
        tuition_val = 1 if "Sí" in tuition else 0
        scholarship = st.selectbox("¿Tiene beca?", ["No (0)", "Sí (1)"])
        scholarship_val = 1 if "Sí" in scholarship else 0

    with col3:
        st.markdown("**🌍 Contexto Socioeconómico**")
        unemployment = st.slider("Tasa de desempleo (%)", 7.0, 17.0, 11.0, 0.1)
        inflation = st.slider("Tasa de inflación (%)", -1.0, 4.0, 1.4, 0.1)
        gdp = st.slider("PIB (variación)", -4.0, 4.0, 1.74, 0.01)
        displaced = st.selectbox("¿Desplazado?", ["No (0)", "Sí (1)"])
        displaced_val = 1 if "Sí" in displaced else 0
        international = st.selectbox("¿Internacional?", ["No (0)", "Sí (1)"])
        international_val = 1 if "Sí" in international else 0

    if st.button("🔮 Calcular Probabilidad de Deserción", type="primary"):
        # Build a sample row using dataset medians for missing features
        sample = df.median(numeric_only=True).to_dict()
        sample.update({
            "Age at enrollment": age,
            "Gender": gender_val,
            "Debtor": debtor_val,
            "Tuition fees up to date": tuition_val,
            "Scholarship holder": scholarship_val,
            "Displaced": displaced_val,
            "International": international_val,
            "Curricular units 1st sem (grade)": grade_1,
            "Curricular units 1st sem (approved)": approved_1,
            "Curricular units 1st sem (enrolled)": enrolled_1,
            "Curricular units 2nd sem (grade)": grade_2,
            "Curricular units 2nd sem (approved)": approved_2,
            "Curricular units 2nd sem (enrolled)": enrolled_2,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
        })
        feat_order = df.drop(columns=["Target_binary"]).columns.tolist()
        X_input = pd.DataFrame([sample])[feat_order]
        X_input_sc = models["scaler_nn"].transform(X_input)
        prob = float(models["model"].predict(X_input_sc, verbose=0)[0][0])

        st.markdown("---")
        st.markdown("### Resultado")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            risk_label = "🔴 ALTO RIESGO" if prob > 0.6 else "🟡 RIESGO MEDIO" if prob > 0.35 else "🟢 BAJO RIESGO"
            css_class = "risk-high" if prob > 0.6 else "risk-medium" if prob > 0.35 else "risk-low"
            st.markdown(f"""
            <div class="{css_class}">
                <h2 style="margin:0">{risk_label}</h2>
                <h1 style="margin:4px 0;font-size:3rem">{prob*100:.1f}%</h1>
                <p style="margin:0">probabilidad de deserción</p>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            fig, ax = plt.subplots(figsize=(4, 4))
            wedge_color = PALETTE[1] if prob > 0.6 else PALETTE[3] if prob > 0.35 else PALETTE[2]
            ax.pie([prob, 1-prob], colors=[wedge_color, "#e0e0e0"],
                   startangle=90, counterclock=False,
                   wedgeprops={"width": 0.5, "edgecolor": "white"})
            ax.text(0, 0, f"{prob*100:.0f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=wedge_color)
            ax.set_title("Probabilidad estimada", fontweight="bold")
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("**Factores de riesgo identificados en este perfil:**")
        factors = []
        if grade_1 < 8: factors.append(f"⚠️ Calificación baja en Sem 1 ({grade_1:.1f})")
        if grade_2 < 8: factors.append(f"⚠️ Calificación baja en Sem 2 ({grade_2:.1f})")
        if approved_1 < 3: factors.append(f"⚠️ Pocas materias aprobadas Sem 1 ({approved_1})")
        if debtor_val == 1: factors.append("⚠️ Estudiante con adeudo")
        if tuition_val == 0: factors.append("⚠️ Colegiatura NO al corriente")
        if age > 25: factors.append(f"⚠️ Edad de ingreso mayor (> 25 años): {age}")
        if not factors:
            st.success("✅ No se detectaron factores de riesgo críticos en este perfil.")
        else:
            for f in factors:
                st.warning(f)


# ══════════════════════════════════════════════
# PÁGINA 6 — RECOMENDACIONES
# ══════════════════════════════════════════════
elif page == "💡 Recomendaciones":
    st.markdown('<p class="section-title">💡 Recomendaciones de Negocio</p>', unsafe_allow_html=True)

    st.markdown("#### Mapa de Riesgo — Priorización de Intervenciones")
    clusters_data = {
        "Rezago académico":      {"riesgo": 82.7, "n": 851},
        "Apoyo familiar sólido": {"riesgo": 29.9, "n": 107},
        "Progreso avanzado":     {"riesgo": 25.2, "n": 282},
        "Rendimiento académico": {"riesgo": 19.3, "n": 3184},
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for nombre, datos in clusters_data.items():
        color = CLUSTER_COLORS[nombre]
        ax.scatter(datos["n"], datos["riesgo"], s=datos["n"]/4,
                   color=color, alpha=0.75, edgecolors="white", linewidth=1.5)
        ax.annotate(nombre, (datos["n"], datos["riesgo"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight="bold")
    avg = df["Target_binary"].mean() * 100
    ax.axhline(avg, color="gray", linestyle="--", label=f"Promedio ({avg:.1f}%)")
    ax.set_xlabel("Número de estudiantes"); ax.set_ylabel("Tasa de Deserción (%)")
    ax.set_title("Mapa Riesgo × Volumen  (tamaño = cantidad de alumnos)", fontweight="bold")
    ax.legend()
    fig.patch.set_alpha(0); plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("""
<div class="risk-high">
<strong>🔴 PRIORIDAD ALTA — Rezago Académico (851 estudiantes, 82.7% deserción)</strong><br><br>
<b>¿Quiénes son?</b> Calificaciones muy bajas (prom. 2.1 en Sem 1), pocas materias aprobadas (~0.4), 
mayor edad (~26 años), alta proporción de hombres y deudores (19.3%).<br><br>
<b>Acciones recomendadas:</b><br>
• Activar alerta temprana automática cuando un estudiante reprueba >50% de materias en el primer parcial<br>
• Asignar tutoría académica obligatoria en las primeras 4 semanas de detección del riesgo<br>
• Revisar carga académica para estudiantes con edad > 24 años (posiblemente trabajan)<br>
• Coordinar planes de pago preventivos para deudores con bajo desempeño
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="risk-medium">
<strong>🟡 PRIORIDAD MEDIA — Apoyo Familiar Sólido (107 estudiantes, 29.9% deserción)</strong><br><br>
<b>¿Quiénes son?</b> Alta presencia de estudiantes internacionales o desplazados, calificaciones medias (~10.8).<br><br>
<b>Acciones recomendadas:</b><br>
• Fortalecer red de apoyo e integración para estudiantes internacionales<br>
• Implementar programa de mentoría entre pares con estudiantes de alto rendimiento
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="risk-low">
<strong>🟢 PRIORIDAD BAJA — Rendimiento Académico (3,184 estudiantes, 19.3% deserción)</strong><br><br>
<b>¿Quiénes son?</b> La mayoría del alumnado, buen desempeño académico, alta tasa de becas (30.3%).<br><br>
<b>Acciones recomendadas:</b><br>
• Mantener y fortalecer los programas de becas — son el factor protector más importante<br>
• Monitorear posibles caídas de rendimiento en el semestre 2 con alertas automáticas
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Regla de Negocio Sugerida para Intervención Temprana")
    st.code("""
SI (materias_aprobadas_sem1 < 2)
   O (calificacion_sem1 < 5.0)
   O (adeudo_colegiatura = Sí  Y  calificacion_sem1 < 8.0)
ENTONCES → Activar protocolo de retención temprana
    """, language="sql")
    st.info("⚡ Aplicando esta regla se estima intervención oportuna en ~**760 estudiantes de alto riesgo real** (17.2% del total).")
