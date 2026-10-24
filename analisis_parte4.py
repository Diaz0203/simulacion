import numpy as np, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from modelo_referencia_simpy import correr, ESTACIONES
sim, det = correr(95, 10, semilla0=5000, detalle=True)
n=len(sim); m=sim.mean(); s=sim.std(ddof=1); h=stats.t.ppf(0.95,n-1)*s/math.sqrt(n)
print(f"IC90 media: {m:.1f} ± {h:.1f} -> [{m-h:.1f}, {m+h:.1f}]  | percentil 90: {np.percentile(sim,90):.1f}  | PI90 [{m-stats.t.ppf(.95,n-1)*s*math.sqrt(1+1/n):.1f}, {m+stats.t.ppf(.95,n-1)*s*math.sqrt(1+1/n):.1f}]")
for nc in (10,18):
    _, d = correr(60, nc, semilla0=7000, detalle=True)
    print(nc, {k: round(100*np.mean([f[k][2] for f in d]),1) for k in [e[0] for e in ESTACIONES]})
ns=[4,6,8,10,12,14,16,18,20,22,24,26,28,30]; med=[]; hw=[]
for nc in ns:
    t,_=correr(100,nc,semilla0=20000); med.append(t.mean()); hw.append(stats.t.ppf(.975,99)*t.std(ddof=1)/10)
    print(nc, round(t.mean(),1), round(hw[-1],1))
fig,ax=plt.subplots(figsize=(8,4.5))
ax.errorbar(ns,med,yerr=hw,fmt="-o",color="#2c6e8f",capsize=3)
ax.set_xlabel("Contenedores asignados"); ax.set_ylabel("Hora de terminación promedio (min)")
ax.set_title("Contenedores vs hora de terminación (100 réplicas, IC 95 %)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("salida_parte4_contenedores.png",dpi=120)
