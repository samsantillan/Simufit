import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import random

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="SimuFit - Los Simuladores",
    layout="wide"
)

st.title("🎲 SimuFit - Ajuste de Datos a Distribuciones de Probabilidad")
st.caption("Aplicación educativa desarrollada por el equipo **Los Simuladores**")

# ============================================================
# BASE TEÓRICA
# ============================================================

DISTRIBUCIONES_INFO = {
    "Bernoulli": {
        "nombre": "Distribución de Bernoulli",
        "que_cuenta": "Un solo ensayo con dos posibles resultados: éxito o fracaso.",
        "valores": "0 o 1",
        "ejemplo": "Un estudiante aprueba o no aprueba; una pieza es defectuosa o no defectuosa.",
        "parametros": "p = probabilidad de éxito.",
        "formula": r"P(X=x)=p^x(1-p)^{1-x}, \quad x \in \{0,1\}",
        "interpretacion": "Si p = 0.70, significa que existe 70% de probabilidad de éxito en un único ensayo."
    },
    "Binomial": {
        "nombre": "Distribución Binomial",
        "que_cuenta": "Número de éxitos en un número fijo de ensayos independientes.",
        "valores": "0, 1, 2, ..., n",
        "ejemplo": "Número de caras al lanzar una moneda 10 veces.",
        "parametros": "n = número de ensayos; p = probabilidad de éxito.",
        "formula": r"P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}",
        "interpretacion": "Si n = 10 y p = 0.50, se cuenta cuántos éxitos aparecen en 10 intentos."
    },
    "Binomial Negativa": {
        "nombre": "Distribución Binomial Negativa",
        "que_cuenta": "Número de fracasos antes de alcanzar una cantidad fija de éxitos.",
        "valores": "0, 1, 2, 3, ...",
        "ejemplo": "Número de intentos fallidos antes de conseguir 3 ventas.",
        "parametros": "r = número de éxitos esperados; p = probabilidad de éxito.",
        "formula": r"P(X=x)=\binom{x+r-1}{x}p^r(1-p)^x",
        "interpretacion": "Si r = 3 y X = 10, significa que hubo 10 fracasos antes de conseguir 3 éxitos."
    },
    "Geométrica": {
        "nombre": "Distribución Geométrica",
        "que_cuenta": "Número de intentos necesarios hasta obtener el primer éxito.",
        "valores": "1, 2, 3, 4, ...",
        "ejemplo": "Número de lanzamientos hasta obtener el primer 6 en un dado.",
        "parametros": "p = probabilidad de éxito.",
        "formula": r"P(X=x)=(1-p)^{x-1}p, \quad x=1,2,3,\ldots",
        "interpretacion": "Si X = 8, significa que el primer éxito ocurrió en el intento 8."
    },
    "Hipergeométrica": {
        "nombre": "Distribución Hipergeométrica",
        "que_cuenta": "Número de éxitos al extraer elementos sin reemplazo de una población finita.",
        "valores": "Depende de M, K y n.",
        "ejemplo": "Seleccionar productos de un lote sin devolverlos y contar cuántos son defectuosos.",
        "parametros": "M = tamaño de población; K = éxitos en la población; n = tamaño de muestra.",
        "formula": r"P(X=x)=\frac{\binom{K}{x}\binom{M-K}{n-x}}{\binom{M}{n}}",
        "interpretacion": "Se usa cuando se extrae sin reemplazo, por eso las probabilidades cambian después de cada extracción."
    },
    "Poisson": {
        "nombre": "Distribución de Poisson",
        "que_cuenta": "Número de eventos que ocurren en un intervalo fijo de tiempo o espacio.",
        "valores": "0, 1, 2, 3, ...",
        "ejemplo": "Número de llamadas por hora, clientes por minuto o accidentes por día.",
        "parametros": "λ = promedio de ocurrencias en el intervalo.",
        "formula": r"P(X=x)=\frac{\lambda^x e^{-\lambda}}{x!}",
        "interpretacion": "Si λ = 4, significa que en promedio ocurren 4 eventos por intervalo."
    }
}

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_aic(loglik, k):
    return 2 * k - 2 * loglik


def calcular_bic(loglik, k, n):
    return k * np.log(n) - 2 * loglik


def prueba_chi_cuadrado_agrupada(data, dist, soporte_min, soporte_max, k_parametros):
    """
    Prueba Chi-cuadrado aproximada para distribuciones discretas.
    Agrupa categorías con frecuencia esperada pequeña para mejorar la validez.
    """
    xs = np.arange(soporte_min, soporte_max + 1)
    obs = np.array([np.sum(data == x) for x in xs], dtype=float)
    exp = np.array([dist.pmf(x) * len(data) for x in xs], dtype=float)

    exp = np.maximum(exp, 1e-12)
    exp = exp * (obs.sum() / exp.sum())

    grupos_obs = []
    grupos_exp = []

    acum_obs = 0
    acum_exp = 0

    for o, e in zip(obs, exp):
        acum_obs += o
        acum_exp += e

        if acum_exp >= 5:
            grupos_obs.append(acum_obs)
            grupos_exp.append(acum_exp)
            acum_obs = 0
            acum_exp = 0

    if acum_exp > 0:
        if len(grupos_obs) > 0:
            grupos_obs[-1] += acum_obs
            grupos_exp[-1] += acum_exp
        else:
            grupos_obs.append(acum_obs)
            grupos_exp.append(acum_exp)

    grupos_obs = np.array(grupos_obs)
    grupos_exp = np.array(grupos_exp)

    if len(grupos_obs) < 2:
        return np.nan, np.nan, np.nan

    chi2 = np.sum((grupos_obs - grupos_exp) ** 2 / grupos_exp)
    grados_libertad = len(grupos_obs) - 1 - k_parametros

    if grados_libertad <= 0:
        return chi2, np.nan, grados_libertad

    pvalor = stats.chi2.sf(chi2, grados_libertad)
    return chi2, pvalor, grados_libertad


# ============================================================
# FUNCIONES DE AJUSTE
# ============================================================

def ajustar_bernoulli(data):
    if not np.all(np.isin(data, [0, 1])):
        return None

    p = np.mean(data)
    dist = stats.bernoulli(p)
    loglik = np.sum(dist.logpmf(data))
    k = 1

    chi2, pvalor, gl = prueba_chi_cuadrado_agrupada(data, dist, 0, 1, k)

    return {
        "Distribución": "Bernoulli",
        "Parámetros": f"p={p:.4f}",
        "AIC": calcular_aic(loglik, k),
        "BIC": calcular_bic(loglik, k, len(data)),
        "LogLik": loglik,
        "Chi2": chi2,
        "Chi2 p-valor": pvalor,
        "gl": gl,
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

    chi2, pvalor, gl = prueba_chi_cuadrado_agrupada(data, dist, soporte_min, soporte_max, k)

    return {
        "Distribución": "Poisson",
        "Parámetros": f"λ={lam:.4f}",
        "AIC": calcular_aic(loglik, k),
        "BIC": calcular_bic(loglik, k, len(data)),
        "LogLik": loglik,
        "Chi2": chi2,
        "Chi2 p-valor": pvalor,
        "gl": gl,
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

    chi2, pvalor, gl = prueba_chi_cuadrado_agrupada(data, dist, soporte_min, soporte_max, k)

    return {
        "Distribución": "Geométrica",
        "Parámetros": f"p={p:.4f}",
        "AIC": calcular_aic(loglik, k),
        "BIC": calcular_bic(loglik, k, len(data)),
        "LogLik": loglik,
        "Chi2": chi2,
        "Chi2 p-valor": pvalor,
        "gl": gl,
        "dist": dist,
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_binomial(data):
    max_x = int(np.max(data))
    media = np.mean(data)

    if max_x < 1:
        return None

    mejor = None

    n_min = max_x
    n_max = max(max_x + 30, int(max_x * 3), 10)

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

    chi2, pvalor, gl = prueba_chi_cuadrado_agrupada(data, mejor["dist"], soporte_min, soporte_max, k)

    return {
        "Distribución": "Binomial",
        "Parámetros": f"n={mejor['n']}, p={mejor['p']:.4f}",
        "AIC": calcular_aic(mejor["LogLik"], k),
        "BIC": calcular_bic(mejor["LogLik"], k, len(data)),
        "LogLik": mejor["LogLik"],
        "Chi2": chi2,
        "Chi2 p-valor": pvalor,
        "gl": gl,
        "dist": mejor["dist"],
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_binomial_negativa(data):
    media = np.mean(data)

    if media <= 0:
        return None

    mejor = None

    for r in range(1, 101):
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

    chi2, pvalor, gl = prueba_chi_cuadrado_agrupada(data, mejor["dist"], soporte_min, soporte_max, k)

    return {
        "Distribución": "Binomial Negativa",
        "Parámetros": f"r={mejor['r']}, p={mejor['p']:.4f}",
        "AIC": calcular_aic(mejor["LogLik"], k),
        "BIC": calcular_bic(mejor["LogLik"], k, len(data)),
        "LogLik": mejor["LogLik"],
        "Chi2": chi2,
        "Chi2 p-valor": pvalor,
        "gl": gl,
        "dist": mejor["dist"],
        "soporte_min": soporte_min,
        "soporte_max": soporte_max
    }


def ajustar_hipergeometrica(data):
    max_x = int(np.max(data))

    if max_x < 1:
        return None

    mejor = None

    M_min = max(max_x + 1, 2)
    M_max = min(max(max_x + 40, 40), 120)

    for M in range(M_min, M_max + 1):
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

    chi2, pvalor, gl = prueba_chi_cuadrado_agrupada(
        data,
        mejor["dist"],
        mejor["soporte_min"],
        mejor["soporte_max"],
        k
    )

    return {
        "Distribución": "Hipergeométrica",
        "Parámetros": f"M={mejor['M']}, K={mejor['K']}, n={mejor['n']}",
        "AIC": calcular_aic(mejor["LogLik"], k),
        "BIC": calcular_bic(mejor["LogLik"], k, len(data)),
        "LogLik": mejor["LogLik"],
        "Chi2": chi2,
        "Chi2 p-valor": pvalor,
        "gl": gl,
        "dist": mejor["dist"],
        "soporte_min": mejor["soporte_min"],
        "soporte_max": mejor["soporte_max"]
    }


# ============================================================
# PESTAÑAS DE LA APP
# ============================================================

tab_inicio, tab_analisis, tab_aprende, tab_juego = st.tabs([
    "🏠 Inicio",
    "📊 Analizar datos",
    "📚 Guía educativa",
    "🎮 Mini juego"
])

# ============================================================
# TAB INICIO
# ============================================================

with tab_inicio:
    st.header("Bienvenido a SimuFit")

    st.markdown("""
    **SimuFit** es una aplicación que permite ingresar datos y determinar qué distribución de probabilidad discreta
    representa mejor su comportamiento.

    La app compara seis distribuciones:

    - Bernoulli
    - Binomial
    - Binomial Negativa
    - Geométrica
    - Hipergeométrica
    - Poisson

    El objetivo no es solo entregar un resultado, sino también explicar **qué significa cada distribución**,
    **cómo se interpreta el ajuste** y **qué decisión estadística se toma**.
    """)

    st.subheader("¿Qué entrega la aplicación?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **1. Ranking de ajuste**

        Ordena las distribuciones según el AIC.

        La mejor distribución es la que tiene el menor AIC.
        """)

    with col2:
        st.success("""
        **2. Prueba de hipótesis**

        Plantea H₀ y H₁.

        Usa el p-valor de Chi-cuadrado para decidir si se rechaza o no H₀.
        """)

    with col3:
        st.warning("""
        **3. Explicación educativa**

        Explica qué representa la distribución ganadora, sus parámetros y su fórmula.
        """)

    st.subheader("Criterios usados por SimuFit")

    st.markdown("""
    - **AIC:** criterio principal para escoger la mejor distribución. Menor AIC significa mejor ajuste relativo.
    - **BIC:** criterio de respaldo. También se prefiere el menor valor, pero penaliza más los modelos complejos.
    - **Log-verosimilitud:** mide qué tan bien una distribución explica los datos. Mientras mayor sea, mejor.
    - **p-valor Chi-cuadrado:** permite evaluar si los datos son compatibles con la distribución propuesta.
    """)

# ============================================================
# TAB ANÁLISIS
# ============================================================

with tab_analisis:
    st.header("📊 Análisis de datos")

    st.subheader("1️⃣ Generar datos aleatorios de prueba")

    with st.expander("Generador de datos", expanded=False):
        col_gen1, col_gen2 = st.columns(2)

        with col_gen1:
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

        with col_gen2:
            if dist_gen == "Bernoulli":
                p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.50)

            elif dist_gen == "Binomial":
                n_bin = st.number_input("n: número de ensayos", min_value=1, value=10)
                p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.50)

            elif dist_gen == "Binomial Negativa":
                r = st.number_input("r: número de éxitos esperados", min_value=1, value=3)
                p = st.slider("p: probabilidad de éxito", 0.01, 0.99, 0.30)

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
            st.session_state["origen"] = f"Datos generados desde {dist_gen}"
            st.success("Datos generados correctamente. Baja a la sección de resultados.")

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
            st.session_state["origen"] = "Datos cargados desde archivo"

        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()

    elif texto.strip():
        try:
            texto_limpio = texto.replace("\n", ",").replace(" ", ",")
            valores = [x.strip() for x in texto_limpio.split(",") if x.strip() != ""]
            data = np.array([float(x.replace(",", ".")) for x in valores])
            st.session_state["origen"] = "Datos pegados manualmente"

        except Exception:
            st.error("Formato inválido. Ingresa solo números separados por comas, espacios o saltos de línea.")
            st.stop()

    elif "data" in st.session_state:
        data = st.session_state["data"]

    else:
        st.info("Primero genera, carga o pega datos para iniciar el análisis.")
        st.stop()

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
        st.dataframe(pd.DataFrame({"value": data}), use_container_width=True)

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
    mejor = resultados[0]

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

    col_tabla, col_grafico = st.columns([1.15, 1])

    with col_tabla:
        st.markdown("### Ranking de ajuste")
        st.dataframe(df_resultados, use_container_width=True)
        st.caption("La distribución ubicada en primer lugar es la de menor AIC.")

        with st.expander("¿Cómo se interpreta esta tabla?"):
            st.markdown("""
            - **AIC:** es el criterio principal. La mejor distribución es la que tiene el menor AIC.
            - **BIC:** sirve como criterio de respaldo. También se prefiere el valor más bajo.
            - **Log-verosimilitud:** indica qué tan bien el modelo explica los datos. Mientras mayor sea, mejor.
            - **Chi2 p-valor:** permite decidir si los datos son compatibles con la distribución propuesta.
            """)

    with col_grafico:
        st.markdown("### Datos observados vs distribución ganadora")

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

    st.markdown("---")
    st.subheader("5️⃣ Conclusión automática e hipótesis")

    mejor_nombre = mejor["Distribución"]
    pvalor = mejor["Chi2 p-valor"]

    st.markdown(f"""
    **Distribución ganadora:** {mejor_nombre}  
    **Parámetros estimados:** {mejor["Parámetros"]}

    **Hipótesis nula H₀:** Los datos siguen una distribución {mejor_nombre}.  
    **Hipótesis alternativa H₁:** Los datos no siguen una distribución {mejor_nombre}.
    """)

    if not np.isnan(pvalor):
        if pvalor > 0.05:
            st.success(f"""
            Como el p-valor de Chi-cuadrado es **{pvalor:.4f}**, y es mayor que 0.05,
            **no se rechaza H₀**.

            Por lo tanto, los datos son compatibles con una distribución **{mejor_nombre}**.
            """)
        else:
            st.warning(f"""
            Como el p-valor de Chi-cuadrado es **{pvalor:.4f}**, y es menor o igual que 0.05,
            **se rechaza H₀**.

            Esto indica que el ajuste debe interpretarse con cautela. Aunque la distribución tuvo el menor AIC,
            la prueba Chi-cuadrado sugiere que los datos no se ajustan perfectamente a esa distribución.
            """)
    else:
        st.info("""
        El p-valor no está disponible porque no existen suficientes grupos válidos para aplicar la prueba Chi-cuadrado.
        En este caso, se interpreta principalmente el AIC y el BIC.
        """)

    st.markdown("---")
    st.subheader("6️⃣ Explicación de la distribución ganadora")

    info = DISTRIBUCIONES_INFO[mejor_nombre]

    col_info1, col_info2 = st.columns([1.4, 1])

    with col_info1:
        st.markdown(f"### {info['nombre']}")
        st.write(f"**¿Qué cuenta?:** {info['que_cuenta']}")
        st.write(f"**Valores posibles:** {info['valores']}")
        st.write(f"**Ejemplo:** {info['ejemplo']}")
        st.write(f"**Interpretación:** {info['interpretacion']}")

    with col_info2:
        st.markdown("### Fórmula")
        st.latex(info["formula"])
        st.write(f"**Parámetros:** {info['parametros']}")

    st.markdown("---")
    st.subheader("7️⃣ Descargar resultados")

    csv = df_resultados.to_csv(index=True).encode("utf-8")

    st.download_button(
        label="📥 Descargar ranking en CSV",
        data=csv,
        file_name="resultados_simufit.csv",
        mime="text/csv"
    )

# ============================================================
# TAB GUÍA EDUCATIVA
# ============================================================

with tab_aprende:
    st.header("📚 Guía educativa")

    st.subheader("Resumen de distribuciones")

    resumen = pd.DataFrame([
        {
            "Distribución": nombre,
            "¿Qué cuenta?": info["que_cuenta"],
            "Valores posibles": info["valores"],
            "Ejemplo": info["ejemplo"]
        }
        for nombre, info in DISTRIBUCIONES_INFO.items()
    ])

    st.dataframe(resumen, use_container_width=True)

    st.subheader("Conceptos clave")

    with st.expander("¿Qué es AIC?"):
        st.markdown("""
        El **AIC** es el Criterio de Información de Akaike.

        Sirve para comparar distribuciones.  
        **Mientras menor sea el AIC, mejor es el ajuste relativo.**

        La app usa el AIC como criterio principal para escoger la distribución ganadora.
        """)

    with st.expander("¿Qué es BIC?"):
        st.markdown("""
        El **BIC** es el Criterio de Información Bayesiano.

        También sirve para comparar modelos.  
        **Mientras menor sea el BIC, mejor.**

        A diferencia del AIC, el BIC penaliza más a los modelos con muchos parámetros.
        """)

    with st.expander("¿Qué es la log-verosimilitud?"):
        st.markdown("""
        La **log-verosimilitud** mide qué tan probable es observar los datos bajo una distribución.

        **Mientras mayor sea la log-verosimilitud, mejor.**

        Como normalmente aparece con números negativos, se interpreta así:

        - -30 es mejor que -80.
        - -80 es mejor que -150.
        """)

    with st.expander("¿Qué es el p-valor Chi-cuadrado?"):
        st.markdown("""
        El **p-valor Chi-cuadrado** ayuda a evaluar si los datos observados son compatibles
        con la distribución propuesta.

        Se usa normalmente un nivel de significancia de 0.05.

        - Si p-valor > 0.05: no se rechaza H₀.
        - Si p-valor ≤ 0.05: se rechaza H₀.
        """)

    with st.expander("¿Cuál es la hipótesis de la prueba?"):
        st.markdown("""
        Para cada distribución ganadora, la app plantea:

        **H₀:** Los datos siguen la distribución propuesta.  
        **H₁:** Los datos no siguen la distribución propuesta.

        Ejemplo:

        Si gana Poisson:

        **H₀:** Los datos siguen una distribución Poisson.  
        **H₁:** Los datos no siguen una distribución Poisson.
        """)

# ============================================================
# TAB MINI JUEGO
# ============================================================

with tab_juego:
    st.header("🎮 Mini juego: identifica la distribución")

    st.markdown("""
    En este juego se muestra una situación o conjunto de datos.  
    Tu objetivo es escoger qué distribución representa mejor el caso.
    """)

    preguntas = [
        {
            "situacion": "Datos: 0, 1, 1, 0, 1, 0. Cada valor representa fracaso o éxito en un solo intento.",
            "respuesta": "Bernoulli",
            "explicacion": "Es Bernoulli porque cada dato solo puede ser 0 o 1."
        },
        {
            "situacion": "Datos: 3, 5, 4, 6, 2. Cada valor representa el número de éxitos obtenidos en 10 ensayos.",
            "respuesta": "Binomial",
            "explicacion": "Es Binomial porque se cuentan éxitos dentro de un número fijo de ensayos."
        },
        {
            "situacion": "Datos: 0, 2, 4, 1, 3. Representan el número de llamadas recibidas por minuto.",
            "respuesta": "Poisson",
            "explicacion": "Es Poisson porque se cuentan eventos en un intervalo fijo de tiempo."
        },
        {
            "situacion": "Datos: 1, 3, 8, 2, 5. Representan el número de intentos hasta lograr el primer éxito.",
            "respuesta": "Geométrica",
            "explicacion": "Es Geométrica porque se cuenta cuántos intentos pasan hasta el primer éxito."
        },
        {
            "situacion": "Datos: 4, 10, 2, 7, 15. Representan fracasos antes de lograr 3 éxitos.",
            "respuesta": "Binomial Negativa",
            "explicacion": "Es Binomial Negativa porque se cuentan fracasos antes de alcanzar r éxitos."
        },
        {
            "situacion": "Datos: 1, 2, 0, 3, 2. Se extraen productos de un lote sin reemplazo y se cuentan defectuosos.",
            "respuesta": "Hipergeométrica",
            "explicacion": "Es Hipergeométrica porque la extracción se realiza sin reemplazo."
        }
    ]

    if "pregunta_actual" not in st.session_state:
        st.session_state["pregunta_actual"] = random.choice(preguntas)
        st.session_state["respondido"] = False

    pregunta = st.session_state["pregunta_actual"]

    st.info(pregunta["situacion"])

    opciones = [
        "Bernoulli",
        "Binomial",
        "Binomial Negativa",
        "Geométrica",
        "Hipergeométrica",
        "Poisson"
    ]

    respuesta_usuario = st.radio("Selecciona la distribución:", opciones)

    colj1, colj2 = st.columns(2)

    with colj1:
        if st.button("✅ Verificar respuesta"):
            st.session_state["respondido"] = True

            if respuesta_usuario == pregunta["respuesta"]:
                st.success(f"Correcto. La respuesta es {pregunta['respuesta']}.")
            else:
                st.error(f"No es correcto. La respuesta correcta es {pregunta['respuesta']}.")

            st.write(pregunta["explicacion"])

    with colj2:
        if st.button("🔄 Nueva pregunta"):
            st.session_state["pregunta_actual"] = random.choice(preguntas)
            st.session_state["respondido"] = False
            st.rerun()
