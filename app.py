import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="SimuFit - Los Simuladores",
    layout="wide"
)

st.title("🎲 SimuFit - Ajuste de Datos a Distribuciones de Probabilidad")
st.caption("Aplicación desarrollada por el equipo **Los Simuladores**")

# ============================================================
# INFORMACIÓN TEÓRICA DE LAS DISTRIBUCIONES
# ============================================================

DISTRIBUCIONES_INFO = {
    "Bernoulli": {
        "nombre": "Distribución de Bernoulli",
        "descripcion": "Modela un experimento con solo dos resultados posibles: éxito o fracaso.",
        "representa": "Cada dato debe ser 0 o 1. Por ejemplo: aprobó/no aprobó, compró/no compró, defectuoso/no defectuoso.",
        "parametros": "p = probabilidad de éxito.",
        "formula": r"P(X=x)=p^x(1-p)^{1-x}, \quad x \in \{0,1\}",
        "supuesto": "Los datos deben estar codificados como 0 y 1."
    },
    "Binomial": {
        "nombre": "Distribución Binomial",
        "descripcion": "Cuenta el número de éxitos en un número fijo de ensayos independientes.",
        "representa": "Por ejemplo: número de caras en 10 lanzamientos o número de productos defectuosos en una muestra.",
        "parametros": "n = número de ensayos; p = probabilidad de éxito.",
        "formula": r"P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}",
        "supuesto": "Debe existir un número fijo de ensayos n y la probabilidad de éxito debe mantenerse constante."
    },
    "Binomial Negativa": {
        "nombre": "Distribución Binomial Negativa",
        "descripcion": "Modela el número de fracasos antes de alcanzar una cantidad fija de éxitos.",
        "representa": "Por ejemplo: número de intentos fallidos antes de conseguir 3 ventas.",
        "parametros": "r = número de éxitos esperados; p = probabilidad de éxito.",
        "formula": r"P(X=x)=\binom{x+r-1}{x}p^r(1-p)^x",
        "supuesto": "Los datos representan conteos de fracasos antes de alcanzar r éxitos."
    },
    "Geométrica": {
        "nombre": "Distribución Geométrica",
        "descripcion": "Modela el número de intentos necesarios hasta obtener el primer éxito.",
        "representa": "Por ejemplo: intentos hasta lograr la primera venta o lanzamientos hasta obtener el primer 6.",
        "parametros": "p = probabilidad de éxito.",
        "formula": r"P(X=x)=(1-p)^{x-1}p, \quad x=1,2,3,\ldots",
        "supuesto": "Los datos deben empezar en 1, porque representan intentos hasta el primer éxito."
    },
    "Hipergeométrica": {
        "nombre": "Distribución Hipergeométrica",
        "descripcion": "Modela éxitos al extraer elementos sin reemplazo desde una población finita.",
        "representa": "Por ejemplo: seleccionar productos de un lote sin devolverlos y contar cuántos son defectuosos.",
        "parametros": "M = tamaño de la población; K = éxitos en la población; n = tamaño de la muestra.",
        "formula": r"P(X=x)=\frac{\binom{K}{x}\binom{M-K}{n-x}}{\binom{M}{n}}",
        "supuesto": "Se usa cuando la extracción es sin reemplazo."
    },
    "Poisson": {
        "nombre": "Distribución de Poisson",
        "descripcion": "Modela el número de eventos que ocurren en un intervalo fijo de tiempo o espacio.",
        "representa": "Por ejemplo: llamadas por hora, clientes por minuto, accidentes por día.",
        "parametros": "λ = promedio de ocurrencias por intervalo.",
        "formula": r"P(X=x)=\frac{\lambda^x e^{-\lambda}}{x!}",
        "supuesto": "Los eventos ocurren de forma independiente y con una tasa promedio constante."
    }
}

# ============================================================
# EXPLICACIÓN DE LA APLICACIÓN
# ============================================================

with st.expander("ℹ️ ¿Qué hace esta aplicación?", expanded=True):
    st.markdown("""
    **SimuFit** permite ingresar datos y comparar su ajuste con seis distribuciones discretas de probabilidad:

    - Bernoulli
    - Binomial
    - Binomial Negativa
    - Geométrica
    - Hipergeométrica
    - Poisson

    La aplicación calcula los parámetros de cada distribución, evalúa el ajuste mediante el criterio **AIC** y presenta un ranking.
    La distribución con menor AIC se considera la mejor candidata para representar los datos.

    Además, la app muestra:

    - Tabla resumen con parámetros estimados.
    - Gráfico comparativo entre datos observados y probabilidades teóricas.
    - Conclusión automática.
    - Explicación teórica de la distribución ganadora.
    """)

# ============================================================
# GENERADOR DE DATOS ALEATORIOS
# ============================================================

st.markdown("---")
st.subheader("1️⃣ Generar datos aleatorios de prueba")

with st.expander("Generador de datos", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        dist_gen = st.selectbox(
            "Selecciona la distribución para generar datos",
            [
                "Bernoulli",
                "Binomial",
                "Binomial Negativa",
                "Geométrica",
                "Hipergeométrica",
                "Poisson"
            ]
        )

        cantidad = st.number_input(
            "Cantidad de datos a generar",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

    with col2:
        if dist_gen == "Bernoulli":
            p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.50)

        elif dist_gen == "Binomial":
            n_bin = st.number_input("n: número de ensayos", min_value=1, value=10)
            p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.50)

        elif dist_gen == "Binomial Negativa":
            r = st.number_input("r: número de éxitos esperados", min_value=1, value=5)
            p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.50)

        elif dist_gen == "Geométrica":
            p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.30)

        elif dist_gen == "Hipergeométrica":
            M = st.number_input("M: tamaño de la población", min_value=2, value=50)
            K = st.number_input("K: éxitos en la población", min_value=1, max_value=int(M), value=20)
            n_muestra = st.number_input("n: tamaño de la muestra", min_value=1, max_value=int(M), value=10)

        elif dist_gen == "Poisson":
            lam = st.number_input("λ: promedio de ocurrencias", min_value=0.01, value=4.0)

    if st.button("🎲 Generar datos y analizar"):
        if dist_gen == "Bernoulli":
            datos_generados = stats.bernoulli.rvs(p, size=cantidad)

        elif dist_gen == "Binomial":
            datos_generados = stats.binom.rvs(n_bin, p, size=cantidad)

        elif dist_gen == "Binomial Negativa":
            datos_generados = stats.nbinom.rvs(r, p, size=cantidad)

        elif dist_gen == "Geométrica":
            datos_generados = stats.geom.rvs(p, size=cantidad)

        elif dist_gen == "Hipergeométrica":
            datos_generados = stats.hypergeom.rvs(M, K, n_muestra, size=cantidad)

        elif dist_gen == "Poisson":
            datos_generados = stats.poisson.rvs(lam, size=cantidad)

        st.session_state["data"] = datos_generados
        st.session_state["origen"] = "generados"
        st.success("Datos generados correctamente. Baja a la sección de análisis.")

# ============================================================
# CARGA DE DATOS
# ============================================================

st.markdown("---")
st.subheader("2️⃣ Cargar o pegar tus propios datos")

col_upload, col_paste = st.columns(2)

with col_upload:
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])

with col_paste:
    texto = st.text_area(
        "O pega tus datos separados por comas, espacios o saltos de línea",
        height=120,
        placeholder="Ejemplo: 2, 3, 5, 4, 6, 2, 1"
    )

data = None

if archivo is not None:
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        st.write("Vista previa del archivo:")
        st.dataframe(df.head(), use_container_width=True)

        columna = st.selectbox("Selecciona la columna que deseas analizar", df.columns)
        data = pd.to_numeric(df[columna], errors="coerce").dropna().values
        st.session_state["origen"] = "archivo"

    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

elif texto.strip():
    try:
        texto_limpio = texto.replace("\n", ",").replace(" ", ",")
        valores = [x.strip() for x in texto_limpio.split(",") if x.strip() != ""]
        data = np.array([float(x.replace(",", ".")) for x in valores])
        st.session_state["origen"] = "pegados"

    except Exception:
        st.error("Formato inválido. Ingresa solo números separados por comas, espacios o saltos de línea.")
        st.stop()

elif "data" in st.session_state:
    data = st.session_state["data"]

else:
    st.info("Primero genera, carga o pega datos para iniciar el análisis.")
    st.stop()

# ============================================================
# VALIDACIÓN DE DATOS
# ============================================================

data = np.array(data)
data = data[~np.isnan(data)]

if len(data) < 5:
    st.error("Se necesitan al menos 5 datos válidos para realizar el análisis.")
    st.stop()

if not np.all(np.mod(data, 1) == 0):
    st.error("""
    Esta versión de SimuFit trabaja con distribuciones discretas.
    Por lo tanto, los datos deben ser números enteros: 0, 1, 2, 3, etc.
    """)
    st.stop()

data = data.astype(int)

if np.any(data < 0):
    st.error("Las distribuciones analizadas requieren datos mayores o iguales a cero.")
    st.stop()

st.markdown("---")
st.subheader("3️⃣ Resumen de los datos ingresados")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Cantidad de datos", len(data))
with col_b:
    st.metric("Mínimo", int(np.min(data)))
with col_c:
    st.metric("Máximo", int(np.max(data)))
with col_d:
    st.metric("Media", round(float(np.mean(data)), 4))

with st.expander("Ver datos utilizados"):
    st.write(data)

# ============================================================
# FUNCIONES DE AJUSTE
# ============================================================

def calcular_aic(loglik, k):
    return 2 * k - 2 * loglik


def calcular_bic(loglik, k, n):
    return k * np.log(n) - 2 * loglik


def prueba_chi_cuadrado(data, dist, soporte_min, soporte_max):
    """
    Calcula una prueba chi-cuadrado aproximada.
    Para evitar errores, las frecuencias esperadas se reescalan para que sumen igual que las observadas.
    """
    xs = np.arange(soporte_min, soporte_max + 1)

    obs = np.array([np.sum(data == x) for x in xs])
    exp = np.array([dist.pmf(x) * len(data) for x in xs])

    exp = np.maximum(exp, 1e-10)
    exp = exp * (obs.sum() / exp.sum())

    try:
        chi2, pvalor = stats.chisquare(obs, exp)
        return chi2, pvalor
    except Exception:
        return np.nan, np.nan


def ajustar_bernoulli(data):
    if not np.all(np.isin(data, [0, 1])):
        return None

    p = np.mean(data)
    dist = stats.bernoulli(p)
    loglik = np.sum(dist.logpmf(data))
    k = 1

    chi2, pvalor = prueba_chi_cuadrado(data, dist, 0, 1)

    return {
        "Distribución": "Bernoulli",
        "Parámetros": f"p={p:.4f}",
        "AIC": calcular_aic(loglik, k),
        "BIC": calcular_bic(loglik, k, len(data)),
        "LogLik": loglik,
        "Chi2 p-valor": pvalor,
        "dist": dist,
        "soporte_min": 0,
        "soporte_max": 1
    }


def ajustar_poisson(data):
    lam = np.mean(data)

    if lam <= 0:
        return None

    dist = stats.poisson(lam)
    loglik = np.sum(dist.logpmf(data))
    k = 1

    soporte_min = 0
    soporte_max = max(int(np.max(data)), int(stats.poisson.ppf(0.999, lam)))

    chi2, pvalor = prueba_chi_cuadrado(data, dist, soporte_min, soporte_max)

    return {
        "Distribución": "Poisson",
        "Parámetros": f"λ={lam:.4f}",
        "AIC": calcular_aic(loglik, k),
        "BIC": calcular_bic(loglik, k, len(data)),
        "LogLik": loglik,
        "Chi2 p-valor": pvalor,
        "dist": dist,
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_geometrica(data):
    if np.any(data < 1):
        return None

    p = 1 / np.mean(data)
    p = min(max(p, 0.0001), 0.9999)

    dist = stats.geom(p)
    loglik = np.sum(dist.logpmf(data))
    k = 1

    soporte_min = 1
    soporte_max = max(int(np.max(data)), int(stats.geom.ppf(0.999, p)))

    chi2, pvalor = prueba_chi_cuadrado(data, dist, soporte_min, soporte_max)

    return {
        "Distribución": "Geométrica",
        "Parámetros": f"p={p:.4f}",
        "AIC": calcular_aic(loglik, k),
        "BIC": calcular_bic(loglik, k, len(data)),
        "LogLik": loglik,
        "Chi2 p-valor": pvalor,
        "dist": dist,
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_binomial(data):
    """
    Ajuste binomial:
    Se busca el mejor n posible.
    Para cada n candidato, p se estima como media/n.
    """
    max_x = int(np.max(data))
    media = np.mean(data)

    if max_x < 1:
        return None

    mejor = None

    n_min = max_x
    n_max = max(max_x + 20, int(max_x * 3), 10)

    for n_bin in range(n_min, n_max + 1):
        p = media / n_bin

        if p <= 0 or p >= 1:
            continue

        dist = stats.binom(n_bin, p)
        loglik = np.sum(dist.logpmf(data))

        if np.isfinite(loglik):
            if mejor is None or loglik > mejor["LogLik"]:
                mejor = {
                    "n": n_bin,
                    "p": p,
                    "dist": dist,
                    "LogLik": loglik
                }

    if mejor is None:
        return None

    k = 2
    soporte_min = 0
    soporte_max = mejor["n"]

    chi2, pvalor = prueba_chi_cuadrado(data, mejor["dist"], soporte_min, soporte_max)

    return {
        "Distribución": "Binomial",
        "Parámetros": f"n={mejor['n']}, p={mejor['p']:.4f}",
        "AIC": calcular_aic(mejor["LogLik"], k),
        "BIC": calcular_bic(mejor["LogLik"], k, len(data)),
        "LogLik": mejor["LogLik"],
        "Chi2 p-valor": pvalor,
        "dist": mejor["dist"],
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_binomial_negativa(data):
    """
    SciPy usa nbinom(r, p), donde X representa el número de fracasos antes de r éxitos.
    Se prueba r entero y para cada r se estima p = r / (r + media).
    """
    media = np.mean(data)

    if media <= 0:
        return None

    mejor = None

    for r in range(1, 51):
        p = r / (r + media)

        if p <= 0 or p >= 1:
            continue

        dist = stats.nbinom(r, p)
        loglik = np.sum(dist.logpmf(data))

        if np.isfinite(loglik):
            if mejor is None or loglik > mejor["LogLik"]:
                mejor = {
                    "r": r,
                    "p": p,
                    "dist": dist,
                    "LogLik": loglik
                }

    if mejor is None:
        return None

    k = 2
    soporte_min = 0
    soporte_max = max(int(np.max(data)), int(mejor["dist"].ppf(0.999)))

    chi2, pvalor = prueba_chi_cuadrado(data, mejor["dist"], soporte_min, soporte_max)

    return {
        "Distribución": "Binomial Negativa",
        "Parámetros": f"r={mejor['r']}, p={mejor['p']:.4f}",
        "AIC": calcular_aic(mejor["LogLik"], k),
        "BIC": calcular_bic(mejor["LogLik"], k, len(data)),
        "LogLik": mejor["LogLik"],
        "Chi2 p-valor": pvalor,
        "dist": mejor["dist"],
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_hipergeometrica(data):
    """
    Ajuste aproximado por búsqueda.
    La hipergeométrica necesita M, K y n.
    Si no se conocen, se prueban combinaciones pequeñas y se elige la de mayor log-verosimilitud.
    """
    max_x = int(np.max(data))

    if max_x < 1:
        return None

    mejor = None

    # Rango moderado para que no se demore demasiado
    M_max = 80

    for M in range(max(max_x + 1, 2), M_max + 1):
        for K in range(max_x, M + 1):
            for n_muestra in range(max_x, M + 1):

                soporte_min = max(0, n_muestra - (M - K))
                soporte_max = min(K, n_muestra)

                if np.min(data) < soporte_min or np.max(data) > soporte_max:
                    continue

                dist = stats.hypergeom(M, K, n_muestra)
                loglik = np.sum(dist.logpmf(data))

                if np.isfinite(loglik):
                    if mejor is None or loglik > mejor["LogLik"]:
                        mejor = {
                            "M": M,
                            "K": K,
                            "n": n_muestra,
                            "dist": dist,
                            "LogLik": loglik,
                            "soporte_min": soporte_min,
                            "soporte_max": soporte_max
                        }

    if mejor is None:
        return None

    k = 3

    chi2, pvalor = prueba_chi_cuadrado(
        data,
        mejor["dist"],
        mejor["soporte_min"],
        mejor["soporte_max"]
    )

    return {
        "Distribución": "Hipergeométrica",
        "Parámetros": f"M={mejor['M']}, K={mejor['K']}, n={mejor['n']}",
        "AIC": calcular_aic(mejor["LogLik"], k),
        "BIC": calcular_bic(mejor["LogLik"], k, len(data)),
        "LogLik": mejor["LogLik"],
        "Chi2 p-valor": pvalor,
        "dist": mejor["dist"],
        "soporte_min": mejor["soporte_min"],
        "soporte_max": mejor["soporte_max"]
    }

# ============================================================
# ANÁLISIS
# ============================================================

st.markdown("---")
st.subheader("4️⃣ Ajuste de distribuciones")

with st.spinner("Calculando ajustes..."):
    funciones = [
        ajustar_bernoulli,
        ajustar_binomial,
        ajustar_binomial_negativa,
        ajustar_geometrica,
        ajustar_hipergeometrica,
        ajustar_poisson
    ]

    resultados = []

    for f in funciones:
        r = f(data)
        if r is not None:
            resultados.append(r)

if len(resultados) == 0:
    st.error("No se pudo ajustar ninguna distribución con los datos ingresados.")
    st.stop()

resultados = sorted(resultados, key=lambda x: x["AIC"])

df_resultados = pd.DataFrame([
    {
        "Distribución": r["Distribución"],
        "Parámetros estimados": r["Parámetros"],
        "AIC": round(r["AIC"], 4),
        "BIC": round(r["BIC"], 4),
        "Log-verosimilitud": round(r["LogLik"], 4),
        "Chi2 p-valor": round(r["Chi2 p-valor"], 4) if not np.isnan(r["Chi2 p-valor"]) else "No disponible"
    }
    for r in resultados
])

df_resultados.index = np.arange(1, len(df_resultados) + 1)

col_tabla, col_grafico = st.columns([1.2, 1])

with col_tabla:
    st.markdown("### Ranking de ajuste")
    st.dataframe(df_resultados, use_container_width=True)
    st.caption("La mejor distribución es la que tiene el menor valor de AIC.")

with col_grafico:
    st.markdown("### Datos observados vs distribución ganadora")

    mejor = resultados[0]
    dist = mejor["dist"]

    xs = np.arange(mejor["soporte_min"], mejor["soporte_max"] + 1)

    obs_freq = np.array([np.sum(data == x) for x in xs]) / len(data)
    prob_teorica = dist.pmf(xs)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=xs,
        y=obs_freq,
        name="Frecuencia observada",
        opacity=0.65
    ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=prob_teorica,
        mode="lines+markers",
        name=f"Probabilidad teórica: {mejor['Distribución']}"
    ))

    fig.update_layout(
        xaxis_title="Valor de X",
        yaxis_title="Probabilidad / Frecuencia relativa",
        template="plotly_white",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CONCLUSIÓN AUTOMÁTICA
# ============================================================

st.markdown("---")
st.subheader("5️⃣ Conclusión automática")

mejor_nombre = mejor["Distribución"]
pvalor = mejor["Chi2 p-valor"]

if not np.isnan(pvalor):
    interpretacion_p = (
        "El p-valor es mayor que 0.05, por lo que no se rechaza el ajuste propuesto."
        if pvalor > 0.05
        else "El p-valor es menor o igual que 0.05, por lo que el ajuste debe interpretarse con cautela."
    )
else:
    interpretacion_p = "El p-valor no está disponible para este ajuste."

st.success(f"""
La distribución que mejor representa los datos ingresados es **{mejor_nombre}**.

Esta distribución obtuvo el menor valor de **AIC**, por lo que es la mejor candidata dentro de las distribuciones evaluadas.

**Parámetros estimados:** {mejor["Parámetros"]}

**Interpretación de la prueba Chi-cuadrado:** {interpretacion_p}
""")

# ============================================================
# FICHA TEÓRICA DE LA DISTRIBUCIÓN GANADORA
# ============================================================

st.markdown("---")
st.subheader("6️⃣ Explicación de la distribución ganadora")

info = DISTRIBUCIONES_INFO[mejor_nombre]

col_info1, col_info2 = st.columns([1.4, 1])

with col_info1:
    st.markdown(f"### {info['nombre']}")
    st.write(f"**Descripción:** {info['descripcion']}")
    st.write(f"**Qué representa:** {info['representa']}")
    st.write(f"**Supuesto principal:** {info['supuesto']}")

with col_info2:
    st.markdown("### Fórmula")
    st.latex(info["formula"])
    st.write(f"**Parámetros:** {info['parametros']}")

# ============================================================
# EXPORTACIÓN
# ============================================================

st.markdown("---")
st.subheader("7️⃣ Descargar resultados")

csv = df_resultados.to_csv(index=True).encode("utf-8")

st.download_button(
    label="📥 Descargar ranking en CSV",
    data=csv,
    file_name="resultados_simufit.csv",
    mime="text/csv"
)
