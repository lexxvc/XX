"""
Dashboard Ejecutivo — Predicción de Deserción Estudiantil

Este dashboard carga los resultados ya generados por los notebooks finales.
No reentrena modelos; solo visualiza outputs oficiales del proyecto.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Deserción Estudiantil — Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = /
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"


def load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        st.warning(f"No se encontró el archivo: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def show_image(name: str, caption: Optional[str] = None):
    path = FIGURES / name
    if path.exists():
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"No se encontró la imagen: {path}")


st.title("🎓 Dashboard Ejecutivo — Predicción de Deserción Estudiantil")
st.caption("Visualización ejecutiva basada en los resultados finales del proyecto")

page = st.sidebar.radio(
    "Navegar a:",
    [
        "📊 Resumen Ejecutivo",
        "🤖 Comparación de Modelos",
        "👥 Segmentación de Estudiantes",
        "🔍 Variables Relevantes",
        "💡 Recomendaciones",
    ],
)


if page == "📊 Resumen Ejecutivo":
    st.header("📊 Resumen Ejecutivo")

    executive_summary = load_csv("executive_summary.csv")
    model_metrics = load_csv("model_metrics.csv")

    if not executive_summary.empty:
        st.subheader("Hallazgos principales")
        st.dataframe(executive_summary, use_container_width=True, hide_index=True)

    if not model_metrics.empty:
        best_model = model_metrics.sort_values("F1_Abandono", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor modelo", best_model["Modelo"])
        c2.metric("Accuracy", f"{best_model['Accuracy'] * 100:.1f}%")
        c3.metric("F1 abandono", f"{best_model['F1_Abandono'] * 100:.1f}%")
        c4.metric("AUC ROC", f"{best_model['AUC_ROC']:.3f}")

    st.subheader("Mapa de riesgo")
    show_image("risk_map.png", "Mapa de riesgo por perfil de estudiante")


elif page == "🤖 Comparación de Modelos":
    st.header("🤖 Comparación de Modelos Supervisados")

    model_metrics = load_csv("model_metrics.csv")
    if not model_metrics.empty:
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)

    show_image("model_comparison.png", "Comparación de desempeño entre modelos supervisados")

    col1, col2 = st.columns(2)
    with col1:
        show_image("model_evaluation_neural_network.png", "Evaluación Red Neuronal")
    with col2:
        show_image("model_evaluation_random_forest.png", "Evaluación Random Forest")


elif page == "👥 Segmentación de Estudiantes":
    st.header("👥 Segmentación Exploratoria con K-Means")

    st.markdown(
        """
        K-Means se utiliza como análisis complementario para identificar perfiles de estudiantes
        con características similares. La predicción formal del abandono se realiza con modelos
        supervisados.
        """
    )

    cluster_summary = load_csv("cluster_summary.csv")
    if not cluster_summary.empty:
        st.dataframe(cluster_summary, use_container_width=True)

    show_image("cluster_overview.png", "Tasa de abandono y distribución por perfil")
    show_image("pca_clusters.png", "Mapa PCA 2D de perfiles de estudiantes")


elif page == "🔍 Variables Relevantes":
    st.header("🔍 Variables más Relevantes")

    feature_importance = load_csv("feature_importance.csv")
    if not feature_importance.empty:
        st.dataframe(feature_importance.head(15), use_container_width=True, hide_index=True)

    show_image("feature_importance.png", "Top 15 variables más importantes — Random Forest")


elif page == "💡 Recomendaciones":
    st.header("💡 Recomendaciones de Negocio")

    st.markdown(
        """
        ### Prioridad alta: estudiantes con alto riesgo académico

        - Implementar sistemas de alerta temprana para estudiantes con bajo desempeño académico.
        - Dar seguimiento prioritario a estudiantes con pocas materias aprobadas.
        - Fortalecer tutorías académicas y acompañamiento personalizado.
        - Monitorear estudiantes con adeudos o dificultades relacionadas con pagos escolares.

        ### Prioridad media: estudiantes con señales parciales de riesgo

        - Realizar seguimiento preventivo mediante evaluaciones periódicas.
        - Promover asesorías académicas y orientación estudiantil.
        - Identificar factores externos relacionados con edad, situación familiar o carga académica.

        ### Prioridad baja: estudiantes con buen desempeño

        - Mantener programas de becas y estímulos académicos.
        - Utilizar este grupo como referencia para prácticas asociadas a permanencia estudiantil.
        - Fortalecer programas de reconocimiento y retención institucional.
        """
    )

    st.subheader("Archivos disponibles")
    st.write(f"Figuras: `{FIGURES}`")
    st.write(f"Tablas: `{TABLES}`")
    st.write(f"Modelos: `{MODELS}`")
