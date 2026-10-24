"""
Parte I - Análisis de los datos de entrada
Taller: celda de ensamble de tarjetas electrónicas (Simulación de Sistemas, CECAR)

Para cada variable del archivo Datos_taller_tarjetas_sistemas.xlsx:
  a) análisis descriptivo (resumen, caja con atípicos, histograma)
  b) independencia (dispersión, autocorrelograma, rachas, Ljung-Box)
  c) bondad de ajuste (Chi-cuadrado, K-S, Anderson-Darling) + expresión FlexSim
  d) tabla resumen

Uso:  python parte1_analisis.py Datos_taller_tarjetas_sistemas.xlsx salida/
"""
import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")
ALPHA = 0.05

XLSX = sys.argv[1] if len(sys.argv) > 1 else "Datos_taller_tarjetas_sistemas.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "salida_parte1"
os.makedirs(OUT, exist_ok=True)

VARIABLES = {
    "corte_A": "Corte (CCZ) familia A", "corte_B": "Corte (CCZ) familia B",
    "corte_C": "Corte (CCZ) familia C", "corte_D": "Corte (CCZ) familia D",
    "taladrado_A": "Taladrado (TO) familia A", "taladrado_B": "Taladrado (TO) familia B",
    "taladrado_C": "Taladrado (TO) familia C",
    "fallas_TEF_ataque": "Tiempo entre fallas - Ataque químico",
    "fallas_TR_ataque": "Tiempo de reparación - Ataque químico",
    "fallas_TEF_soldadura": "Tiempo entre fallas - Soldadura",
    "fallas_TR_soldadura": "Tiempo de reparación - Soldadura",
}


# ---------------------------------------------------------------- candidatas
def fit_candidates(x):
    """Ajusta las distribuciones candidatas. Devuelve dict nombre -> (scipy_dist, params, n_param)."""
    c = {}
    c["Normal"] = (stats.norm, stats.norm.fit(x), 2)
    c["Lognormal"] = (stats.lognorm, stats.lognorm.fit(x, floc=0), 2)
    c["Exponencial"] = (stats.expon, stats.expon.fit(x, floc=0), 1)
    c["Gamma"] = (stats.gamma, stats.gamma.fit(x, floc=0), 2)
    c["Weibull"] = (stats.weibull_min, stats.weibull_min.fit(x, floc=0), 2)
    # Erlang = gamma con forma entera k: se elige el k que maximiza la verosimilitud
    best = None
    for k in range(1, 60):
        scale = x.mean() / k
        ll = stats.gamma.logpdf(x, k, 0, scale).sum()
        if best is None or ll > best[0]:
            best = (ll, k, scale)
    c["Erlang"] = (stats.gamma, (best[1], 0, best[2]), 2)
    c["Uniforme"] = (stats.uniform, (x.min(), x.max() - x.min()), 2)
    a, b = x.min(), x.max()
    mode = float(np.clip(3 * x.mean() - a - b, a, b))
    c["Triangular"] = (stats.triang, ((mode - a) / (b - a), a, b - a), 3)
    return c


def flexsim_expr(name, p):
    s = "getstream(current)"
    if name == "Normal":
        return f"normal({p[0]:.4f}, {p[1]:.4f}, {s})"
    if name == "Lognormal":   # lognorm(shape=sigma, loc, scale=exp(mu))
        return f"lognormal2(0, {p[2]:.4f}, {p[0]:.4f}, {s})"
    if name == "Exponencial":
        return f"exponential(0, {p[1]:.4f}, {s})"
    if name == "Gamma":
        return f"gamma(0, {p[2]:.4f}, {p[0]:.4f}, {s})"
    if name == "Weibull":
        return f"weibull(0, {p[2]:.4f}, {p[0]:.4f}, {s})"
    if name == "Erlang":
        return f"erlang(0, {p[2]:.4f}, {int(p[0])}, {s})"
    if name == "Uniforme":
        return f"uniform({p[0]:.4f}, {p[0] + p[1]:.4f}, {s})"
    if name == "Triangular":
        a, w = p[1], p[2]
        return f"triangular({a:.4f}, {a + w:.4f}, {a + p[0] * w:.4f}, {s})"


def libro(name, p):
    if name == "Normal":      return f"N(μ={p[0]:.3f}, σ={p[1]:.3f})"
    if name == "Lognormal":   return f"LN(μ={np.log(p[2]):.3f}, σ={p[0]:.3f})"
    if name == "Exponencial": return f"Exp(media={p[1]:.3f})"
    if name == "Gamma":       return f"Gamma(α={p[0]:.3f}, β={p[2]:.3f})"
    if name == "Weibull":     return f"Weibull(α={p[0]:.3f}, β={p[2]:.3f})"
    if name == "Erlang":      return f"{int(p[0])}-Er({p[0]*p[2]:.3f})"
    if name == "Uniforme":    return f"U({p[0]:.3f}, {p[0]+p[1]:.3f})"
    if name == "Triangular":  return f"T({p[1]:.3f}, {p[1]+p[0]*p[2]:.3f}, {p[1]+p[2]:.3f})"


# ---------------------------------------------------------------- pruebas
def chi2_test(x, dist, p, n_param, k=10):
    """Chi-cuadrado con k clases equiprobables bajo la distribución ajustada (E_i = n/k)."""
    n = len(x)
    edges = dist.ppf(np.linspace(0, 1, k + 1), *p)
    edges[0], edges[-1] = -np.inf, np.inf
    obs = np.histogram(x, bins=edges)[0]
    exp = np.full(k, n / k)
    chi = ((obs - exp) ** 2 / exp).sum()
    gl = k - 1 - n_param
    return chi, gl, stats.chi2.ppf(1 - ALPHA, gl), 1 - stats.chi2.cdf(chi, gl)


def ad_stat(x, dist, p):
    z = np.sort(dist.cdf(x, *p))
    z = np.clip(z, 1e-12, 1 - 1e-12)
    n = len(z)
    i = np.arange(1, n + 1)
    return -n - np.mean((2 * i - 1) * (np.log(z) + np.log(1 - z[::-1])))


AD_CRIT = 2.492   # valor crítico A² (α = 0.05, caso general)


def analizar(hoja, x, titulo):
    n = len(x)
    res = {"variable": hoja, "titulo": titulo, "n": n}
    # ---------------- a) descriptivo
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    lie, lse = q1 - 3 * iqr, q3 + 3 * iqr
    atip = x[(x < li) | (x > ls)]
    extremos = x[(x < lie) | (x > lse)]
    res.update(media=x.mean(), mediana=q2, de=x.std(ddof=1), cv=x.std(ddof=1) / x.mean(),
               minimo=x.min(), maximo=x.max(), q1=q1, q3=q3, iqr=iqr,
               asimetria=stats.skew(x, bias=False), curtosis=stats.kurtosis(x, bias=False),
               lim_inf=li, lim_sup=ls, atipicos=list(np.round(atip, 2)),
               extremos=list(np.round(extremos, 2)))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot(x, vert=False, widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor="#cfe3ee"), flierprops=dict(markerfacecolor="#d9534f"))
    for v in atip:
        ax[0].annotate(f"{v:.2f}", (v, 1), xytext=(0, 12), textcoords="offset points",
                       ha="center", fontsize=8, color="#d9534f")
    ax[0].set_title("Diagrama de caja")
    ax[0].set_xlabel("min")
    k = int(np.ceil(1 + 3.322 * np.log10(n)))   # regla de Sturges
    ax[1].hist(x, bins=k, color="#2c6e8f", edgecolor="white")
    ax[1].set_title(f"Histograma (Sturges, {k} clases)")
    ax[1].set_xlabel("min"); ax[1].set_ylabel("Frecuencia")
    fig.suptitle(titulo)
    fig.tight_layout(); fig.savefig(f"{OUT}/{hoja}_a_descriptivo.png", dpi=120); plt.close(fig)

    # ---------------- b) independencia
    z_rachas, p_rachas = runstest_1samp(x, cutoff="median", correction=False)
    lb = acorr_ljungbox(x, lags=[1, 10], return_df=True)
    q_lb, p_lb = float(lb["lb_stat"].iloc[1]), float(lb["lb_pvalue"].iloc[1])
    p_lb1 = float(lb["lb_pvalue"].iloc[0])
    res["lb1_p"] = p_lb1
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    res.update(rachas_z=z_rachas, rachas_p=p_rachas, lb_q=q_lb, lb_p=p_lb, r1=r1,
               independiente=(p_rachas > ALPHA and p_lb > ALPHA))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(np.arange(1, n + 1), x, "-o", ms=3, color="#2c6e8f")
    ax[0].set_title("Valores en orden de registro"); ax[0].set_xlabel("Observación i")
    ax[1].scatter(x[:-1], x[1:], s=14, color="#2c6e8f")
    ax[1].set_title(f"Dispersión X(i) vs X(i+1)   r1 = {r1:.3f}")
    ax[1].set_xlabel("X(i)"); ax[1].set_ylabel("X(i+1)")
    plot_acf(x, lags=20, ax=ax[2], alpha=0.05)
    ax[2].set_title("Autocorrelograma (bandas 95 %)")
    fig.suptitle(f"{titulo} - Independencia: rachas p = {p_rachas:.3f}, Ljung-Box(10) p = {p_lb:.3f}")
    fig.tight_layout(); fig.savefig(f"{OUT}/{hoja}_b_independencia.png", dpi=120); plt.close(fig)

    # ---------------- c) bondad de ajuste
    filas = []
    for nombre, (dist, p, m) in fit_candidates(x).items():
        chi, gl, chi_crit, chi_p = chi2_test(x, dist, p, m)
        ks, ks_p = stats.kstest(x, dist.cdf, args=p)
        ad = ad_stat(x, dist, p)
        ks_crit = 1.36 / np.sqrt(n)
        acepta = (chi < chi_crit) and (ks < ks_crit) and (ad < AD_CRIT)
        filas.append(dict(distribucion=nombre, parametros=libro(nombre, p), chi2=chi, gl=gl,
                          chi2_crit=chi_crit, chi2_p=chi_p, ks=ks, ks_crit=ks_crit, ks_p=ks_p,
                          ad=ad, ad_crit=AD_CRIT, acepta=acepta,
                          flexsim=flexsim_expr(nombre, p), _dist=dist, _p=p))
    rank = pd.DataFrame(filas)
    for col in ("chi2", "ks", "ad"):
        rank["r_" + col] = rank[col].rank()
    rank["rank_prom"] = rank[["r_chi2", "r_ks", "r_ad"]].mean(axis=1)
    rank = rank.sort_values(["acepta", "rank_prom"], ascending=[False, True]).reset_index(drop=True)
    rank.drop(columns=["_dist", "_p"]).to_csv(f"{OUT}/{hoja}_c_ranking.csv", index=False,
                                              float_format="%.4f")
    aceptadas = rank[rank["acepta"]]
    elegida = (aceptadas if len(aceptadas) else rank).iloc[0]
    res.update(ranking=rank, elegida=elegida["distribucion"], elegida_libro=elegida["parametros"],
               elegida_flexsim=elegida["flexsim"], n_aceptadas=len(aceptadas))

    # gráficos de distribuciones aceptadas (máximo 4)
    mostrar = (aceptadas if len(aceptadas) else rank).head(4)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    xs = np.linspace(max(0, x.min() * 0.8) if x.min() > 0 else x.min(), x.max() * 1.1, 400)
    ax[0].hist(x, bins=k, density=True, color="#cfe3ee", edgecolor="white", label="Datos")
    xe = np.sort(x)
    ax[1].step(xe, np.arange(1, n + 1) / n, where="post", color="black", label="FDA empírica")
    for _, r in mostrar.iterrows():
        lw = 2.6 if r["distribucion"] == elegida["distribucion"] else 1.3
        ax[0].plot(xs, r["_dist"].pdf(xs, *r["_p"]), lw=lw, label=r["distribucion"])
        ax[1].plot(xs, r["_dist"].cdf(xs, *r["_p"]), lw=lw, label=r["distribucion"])
    ax[0].set_title("Densidad ajustada vs histograma"); ax[0].legend(fontsize=8)
    ax[1].set_title("FDA ajustada vs empírica"); ax[1].legend(fontsize=8)
    fig.suptitle(f"{titulo} - seleccionada: {elegida['parametros']}")
    fig.tight_layout(); fig.savefig(f"{OUT}/{hoja}_c_ajuste.png", dpi=120); plt.close(fig)
    return res


def main():
    xl = pd.ExcelFile(XLSX)
    resultados = []
    for hoja, titulo in VARIABLES.items():
        df = xl.parse(hoja)
        x = df.iloc[:, 1].dropna().astype(float).to_numpy()
        r = analizar(hoja, x, titulo)
        resultados.append(r)
        print("=" * 90)
        print(f"{titulo}  (n = {r['n']})")
        print(f"  media {r['media']:.3f} | mediana {r['mediana']:.3f} | DE {r['de']:.3f} | CV {r['cv']:.3f}"
              f" | min {r['minimo']:.2f} | max {r['maximo']:.2f} | asim {r['asimetria']:.3f}"
              f" | curt {r['curtosis']:.3f}")
        print(f"  Q1 {r['q1']:.3f} Q3 {r['q3']:.3f} IQR {r['iqr']:.3f} | límites [{r['lim_inf']:.2f}, "
              f"{r['lim_sup']:.2f}] | atípicos {r['atipicos']} | extremos (3·IQR) {r['extremos']}")
        print(f"  Rachas: Z = {r['rachas_z']:.3f}, p = {r['rachas_p']:.4f} | Ljung-Box(10): "
              f"Q = {r['lb_q']:.3f}, p = {r['lb_p']:.4f} | LB(1) p = {r['lb1_p']:.4f} | r1 = {r['r1']:.3f} "
              f"-> {'INDEPENDIENTES' if r['independiente'] else 'NO INDEPENDIENTES'}")
        cols = ["distribucion", "parametros", "chi2", "chi2_crit", "ks", "ks_crit", "ad", "acepta",
                "rank_prom"]
        print(r["ranking"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"  >> Seleccionada: {r['elegida_libro']}   FlexSim: {r['elegida_flexsim']}")

    resumen = pd.DataFrame([{
        "Variable": r["titulo"], "Hoja": r["variable"], "Media": round(r["media"], 3),
        "DE": round(r["de"], 3), "Atípicos": len(r["atipicos"]),
        "Rachas p": round(r["rachas_p"], 4), "Ljung-Box(10) p": round(r["lb_p"], 4), "Ljung-Box(1) p": round(r["lb1_p"], 4),
        "r1": round(r["r1"], 3),
        "Independiente": "Sí" if r["independiente"] else "No",
        "Distribución": r["elegida_libro"], "Expresión FlexSim": r["elegida_flexsim"],
    } for r in resultados])
    resumen.to_csv(f"{OUT}/tabla_resumen_11_distribuciones.csv", index=False)
    print("\nTABLA RESUMEN")
    print(resumen.to_string(index=False))
    return resultados


if __name__ == "__main__":
    main()
