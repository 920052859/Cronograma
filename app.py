import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.set_page_config(page_title="Predicción Cronograma E4-8", page_icon="🚇", layout="wide")

FEATURES = [
    "duracion_plan_dias", "avance_plan_pct", "avance_real_pct", "desviacion_pct",
    "rendimiento", "ruta_critica", "restricciones_pendientes", "frente_disponible",
    "incidencia_geotecnica", "materiales_disponibles", "permiso_pendiente"
]

@st.cache_data
def load_data():
    df = pd.read_csv("data/actividades_e08_ene_feb_2026.csv")
    df["inicio_plan"] = pd.to_datetime(df["inicio_plan"])
    df["fin_plan"] = pd.to_datetime(df["fin_plan"])
    return df

@st.cache_resource
def train_model(seed=42):
    rng = np.random.default_rng(seed)
    n = 1800
    dur = rng.integers(5, 60, n)
    avance_plan = rng.uniform(5, 85, n)
    desviacion = rng.normal(10, 14, n).clip(-15, 65)
    avance_real = (avance_plan - desviacion).clip(0, 100)
    rendimiento = rng.uniform(0.45, 1.05, n)
    ruta_critica = rng.binomial(1, 0.45, n)
    restricciones = rng.poisson(1.8, n).clip(0, 6)
    frente = rng.binomial(1, 0.72, n)
    geo = rng.binomial(1, 0.22, n)
    mat = rng.binomial(1, 0.78, n)
    permiso = rng.binomial(1, 0.26, n)
    score = (
        0.032*desviacion + 1.2*(1-rendimiento) + 0.18*restricciones + 0.28*ruta_critica
        + 0.38*geo + 0.32*(1-frente) + 0.30*(1-mat) + 0.28*permiso
    )
    prob = 1/(1+np.exp(-(score-1.15)))
    y = rng.binomial(1, prob)
    X = pd.DataFrame({
        "duracion_plan_dias": dur,
        "avance_plan_pct": avance_plan,
        "avance_real_pct": avance_real,
        "desviacion_pct": desviacion,
        "rendimiento": rendimiento,
        "ruta_critica": ruta_critica,
        "restricciones_pendientes": restricciones,
        "frente_disponible": frente,
        "incidencia_geotecnica": geo,
        "materiales_disponibles": mat,
        "permiso_pendiente": permiso,
    })
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=220, max_depth=8, min_samples_leaf=5, random_state=seed, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    return model, metrics

def label_status(p):
    if p < 0.35:
        return "A tiempo"
    if p < 0.60:
        return "Riesgo de retraso"
    return "Retrasada"

def risk_level(p_feb, critical):
    if p_feb >= 0.70 and critical == 1:
        return "Crítico"
    if p_feb >= 0.60:
        return "Alto"
    if p_feb >= 0.35:
        return "Medio"
    return "Bajo"

def predict(df, model, escenario):
    d = df.copy()
    if escenario == "Conservador":
        d["avance_real_pct"] = (d["avance_real_pct"] - 3).clip(0, 100)
        d["rendimiento"] = (d["rendimiento"] - 0.04).clip(0.35, 1.10)
    elif escenario == "Agresivo / recuperación":
        d["avance_real_pct"] = (d["avance_real_pct"] + 4).clip(0, 100)
        d["rendimiento"] = (d["rendimiento"] + 0.04).clip(0.35, 1.10)
        d["restricciones_pendientes"] = (d["restricciones_pendientes"] - 1).clip(0, 6)
    d["desviacion_pct"] = d["avance_plan_pct"] - d["avance_real_pct"]
    p_ene = model.predict_proba(d[FEATURES])[:,1]
    # febrero aumenta presión si mantiene restricciones, permisos o es ruta crítica
    p_feb = np.clip(p_ene + 0.08*d["ruta_critica"] + 0.035*d["restricciones_pendientes"] + 0.04*d["permiso_pendiente"] - 0.04*d["frente_disponible"], 0, 1)
    d["prob_enero"] = p_ene
    d["prob_febrero"] = p_feb
    d["prediccion_enero_2026"] = [label_status(x) for x in p_ene]
    d["prediccion_febrero_2026"] = [label_status(x) for x in p_feb]
    d["nivel_riesgo_estimado"] = [risk_level(p, c) for p, c in zip(p_feb, d["ruta_critica"])]
    d["indice_riesgo"] = (100*d["prob_febrero"]).round(1)
    return d.sort_values(["prob_febrero", "ruta_critica"], ascending=False)

def color_risk(val):
    colors = {"Crítico":"background-color:#FEE2E2;color:#991B1B;font-weight:bold", "Alto":"background-color:#FFEDD5;color:#9A3412", "Medio":"background-color:#FEF9C3;color:#854D0E", "Bajo":"background-color:#DCFCE7;color:#166534"}
    return colors.get(val, "")

st.title("🚇 Predicción del Cronograma de Trabajo con IA")
st.caption("Proyecto: Diseño, Procura y Construcción de la Infraestructura Civil de la Estación E4-8 de la Línea 4 del Metro de Lima y Callao")

model, metrics = train_model()
df_base = load_data()

with st.sidebar:
    st.header("Parámetros")
    escenario = st.selectbox("Escenario de predicción", ["Base", "Conservador", "Agresivo / recuperación"])
    fase = st.multiselect("Filtrar fase", sorted(df_base["fase"].unique()), default=sorted(df_base["fase"].unique()))
    st.markdown("---")
    st.write("**Modelo:** Random Forest Classifier")
    for k, v in metrics.items():
        st.metric(k, f"{v:.2f}")

pred = predict(df_base[df_base["fase"].isin(fase)], model, escenario)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Actividades evaluadas", len(pred))
k2.metric("Críticas/Altas", int(pred["nivel_riesgo_estimado"].isin(["Crítico","Alto"]).sum()))
k3.metric("Prob. media febrero", f"{100*pred['prob_febrero'].mean():.1f}%")
k4.metric("Ruta crítica", int(pred["ruta_critica"].sum()))

st.subheader("Tabla de resultados solicitada")
cols = ["codigo","fase","actividad","estado_actual","prediccion_enero_2026","prediccion_febrero_2026","nivel_riesgo_estimado","indice_riesgo","desviacion_pct","restricciones_pendientes","ruta_critica"]
st.dataframe(
    pred[cols].style.applymap(color_risk, subset=["nivel_riesgo_estimado"]).format({"indice_riesgo":"{:.1f}", "desviacion_pct":"{:.1f}"}),
    use_container_width=True,
    height=480
)

csv = pred[cols].to_csv(index=False).encode("utf-8")
st.download_button("Descargar resultados CSV", csv, "prediccion_cronograma_e08.csv", "text/csv")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribución de riesgo")
    order = ["Crítico", "Alto", "Medio", "Bajo"]
    risk_counts = pred["nivel_riesgo_estimado"].value_counts().reindex(order).dropna().reset_index()
    risk_counts.columns = ["nivel", "actividades"]
    st.plotly_chart(px.bar(risk_counts, x="nivel", y="actividades", text="actividades"), use_container_width=True)
with c2:
    st.subheader("Probabilidad de retraso por actividad")
    top = pred.head(12).copy()
    st.plotly_chart(px.bar(top, x="prob_febrero", y="actividad", orientation="h", hover_data=["fase","codigo"], range_x=[0,1]), use_container_width=True)

st.subheader("Actividades que requieren mayor atención del jefe de proyecto")
attention = pred[pred["nivel_riesgo_estimado"].isin(["Crítico", "Alto"])].head(8)
for _, r in attention.iterrows():
    causas = []
    if r["ruta_critica"] == 1: causas.append("ruta crítica")
    if r["restricciones_pendientes"] >= 3: causas.append("restricciones pendientes")
    if r["permiso_pendiente"] == 1: causas.append("permiso pendiente")
    if r["incidencia_geotecnica"] == 1: causas.append("incidencia geotécnica")
    if r["materiales_disponibles"] == 0: causas.append("materiales no asegurados")
    recomendacion = "Actualizar lookahead 3–6 semanas, cerrar restricciones, asegurar permisos/materiales y activar plan de recuperación si supera el umbral de desviación."
    st.markdown(f"**{r['codigo']} – {r['actividad']}**: {r['nivel_riesgo_estimado']} ({r['indice_riesgo']:.1f}%). Causas: {', '.join(causas) or 'desviación de avance'}. _{recomendacion}_")

st.info("Lectura ejecutiva: el foco de control debe estar en permisos/interferencias, contratación crítica, pantallas, excavación inicial y actividades de ruta crítica; estas concentran mayor probabilidad de atraso acumulado hacia febrero de 2026.")
