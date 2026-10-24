"""
Modelo de referencia (Python + SimPy) de la celda de ensamble de tarjetas.

NO reemplaza el modelo de FlexSim que pide el taller: sirve para tener una idea
de los valores esperados y verificar que el modelo de FlexSim se comporte parecido.
Replica la misma lógica: programa diario aleatorio en t = 0, contenedores que
circulan en una sola banda (sin rebases), capacidades de la Tabla 1, tiempos por
familia, 5 % de reproceso de Inspección a Copia de patrones y fallas en Ataque
químico y Soldadura con las distribuciones obtenidas en la Parte I.

Uso:
  python modelo_referencia_simpy.py                 # piloto + réplicas + análisis
"""
import math
import numpy as np
import simpy

# ------------------------------------------------------------------ datos
# Familias (Tabla C.1): A=1, B=2, C=3, D=4 ; columnas CTC, CCZ, TO, DE, DCC
FAM = {1: (3, 1, 1, 2, 3), 2: (1, 3, 1, 1, 1), 3: (1, 3, 1, 2, 2), 4: (2, 3, 1, 1, 2),
       5: (2, 2, 1, 1, 3), 6: (3, 2, 2, 1, 2), 7: (2, 4, 2, 1, 1), 8: (1, 1, 3, 1, 1),
       9: (1, 1, 3, 1, 2), 10: (2, 1, 3, 1, 3)}


def programa_diario(rng):
    q = {1: rng.integers(7, 12), 2: rng.poisson(8), 3: rng.poisson(4), 4: rng.poisson(2),
         5: rng.poisson(5), 6: rng.integers(12, 19), 7: rng.poisson(5), 8: rng.integers(0, 3),
         9: rng.poisson(8), 10: rng.integers(18, 27)}
    tarjetas = []
    for tipo in range(1, 11):
        tarjetas += [tipo] * int(q[tipo])
    return tarjetas


def erl(rng, media, k):
    return rng.gamma(k, media / k)


def t_carga(rng, tipo):
    f = FAM[tipo][0]
    return {1: erl(rng, 7.55, 3), 2: erl(rng, 7.70, 3), 3: erl(rng, 8.28, 2), 4: erl(rng, 8.82, 3)}[f]


def t_descarga(rng, tipo):
    f = FAM[tipo][4]
    return {1: erl(rng, 9.45, 1), 2: erl(rng, 9.36, 4), 3: erl(rng, 8.12, 2), 4: erl(rng, 8.52, 4)}[f]


def t_deposicion(rng, tipo):
    f = FAM[tipo][3]
    return 1.7 + (erl(rng, 3.12, 3) if f == 1 else erl(rng, 4.44, 3))


# Distribuciones de la Parte I
def t_corte(rng, tipo):
    f = FAM[tipo][1]
    if f == 1:
        return 6.2808 * rng.weibull(2.0090)
    if f == 2:
        return rng.lognormal(math.log(9.8906), 0.3824)
    if f == 3:
        return erl(rng, 6.720, 4)
    return 2.8255 * rng.weibull(0.9725)


def t_taladrado(rng, tipo):
    f = FAM[tipo][2]
    return {1: lambda: erl(rng, 6.253, 7), 2: lambda: erl(rng, 8.534, 33),
            3: lambda: erl(rng, 5.799, 6)}[f]()


FALLAS = {
    "AQ": (lambda r: r.normal(241.2846, 14.6175), lambda r: r.lognormal(math.log(9.1969), 0.5768)),
    "SO": (lambda r: 318.7417 * r.weibull(2.5913), lambda r: erl(r, 14.480, 4)),
}

# nombre, en espera, en proceso, tiempo(rng, tipo), transporte a la siguiente (min)
ESTACIONES = [
    ("CTC", 4, 1, t_carga, 0.5),
    ("LL", 0, 5, lambda r, t: 16.0, 0.1),
    ("CCZ", 6, 1, t_corte, 2.8),
    ("TO", 0, 1, t_taladrado, 2.5),
    ("CP", 0, 1, lambda r, t: 4.5, 3.5),
    ("AQ", 0, 1, lambda r, t: 4.0, 1.2),
    ("DE", 9, 1, t_deposicion, 1.3),
    ("IN", 0, 1, lambda r, t: 5.5, 0.5),
    ("IC", 0, 1, lambda r, t: 6.0, 0.5),
    ("SO", 4, 1, lambda r, t: 6.6, 3.3),
    ("LP", 0, 10, lambda r, t: 7.2, 2.7),
    ("DCC", 9, 1, t_descarga, 0.5),
]
IDX = {e[0]: i for i, e in enumerate(ESTACIONES)}


# ------------------------------------------------------------------ modelo
class Estacion:
    def __init__(self, env, nombre, espera, proceso):
        self.env, self.nombre = env, nombre
        self.cap = espera + proceso
        self.cupos = simpy.Resource(env, capacity=self.cap)      # admisión (espera + proceso)
        self.servidor = simpy.Resource(env, capacity=proceso)    # proceso
        self.proceso = proceso
        self.caida = False
        self.reparada = env.event()
        self.activos = set()
        # estadísticas
        self.n, self.area_n, self.t_ult = 0, 0.0, 0.0
        self.ocupado, self.area_ocup = 0, 0.0
        self.t_falla = 0.0
        self.permanencias = []

    def _acum(self):
        dt = self.env.now - self.t_ult
        self.area_n += self.n * dt
        self.area_ocup += self.ocupado * dt
        self.t_ult = self.env.now

    def entra(self):
        self._acum(); self.n += 1

    def sale(self, t_entrada):
        self._acum(); self.n -= 1
        self.permanencias.append(self.env.now - t_entrada)

    def procesar(self, dur):
        self._acum(); self.ocupado += 1
        resta = dur
        me = self.env.active_process
        while resta > 1e-9:
            if self.caida:
                yield self.reparada
            ini = self.env.now
            self.activos.add(me)
            try:
                yield self.env.timeout(resta)
                resta = 0
            except simpy.Interrupt:
                resta -= self.env.now - ini
            finally:
                self.activos.discard(me)
        self._acum(); self.ocupado -= 1

    def fallas(self, rng, tef, tr):
        while True:
            yield self.env.timeout(max(0.0, tef(rng)))
            self.caida = True
            self.reparada = self.env.event()
            for p in list(self.activos):
                p.interrupt()
            d = max(0.0, tr(rng))
            self.t_falla += d
            yield self.env.timeout(d)
            self.caida = False
            self.reparada.succeed()


class Banda:
    """Tramo de banda: tiempo de recorrido fijo; un contenedor no rebasa a otro."""
    def __init__(self, env, largo):
        self.env, self.largo, self.ult_llegada = env, largo, 0.0

    def recorrer(self):
        llegada = max(self.env.now + self.largo, self.ult_llegada)
        self.ult_llegada = llegada
        yield self.env.timeout(llegada - self.env.now)


def replica(semilla, n_contenedores=10, p_reproceso=0.05):
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()
    est = [Estacion(env, n, e, p) for n, e, p, _, _ in ESTACIONES]
    bandas = [Banda(env, e[4]) for e in ESTACIONES]
    for nom, (tef, tr) in FALLAS.items():
        env.process(est[IDX[nom]].fallas(rng, tef, tr))
    almacen = programa_diario(rng)
    total = len(almacen)
    out = {"termino": 0.0, "hechas": 0, "total": total, "ciclos": []}
    fin = env.event()
    if total == 0:
        return out, est

    def visitar(i, tipo):
        e = est[i]
        req = e.cupos.request()
        yield req                                   # espera en la banda hasta que haya cupo
        t0 = env.now; e.entra()
        with e.servidor.request() as s:
            yield s
            yield from e.procesar(ESTACIONES[i][3](rng, tipo))
        e.sale(t0)
        e.cupos.release(req)

    def contenedor(cid):
        while almacen:
            tipo = almacen.pop(0)                   # toma la siguiente tarjeta
            t_ini = env.now
            i = 0
            while i < len(ESTACIONES):
                yield from visitar(i, tipo)
                if ESTACIONES[i][0] == "IN" and rng.random() < p_reproceso:
                    i = IDX["CP"]                   # reproceso, regreso sin tiempo
                    continue
                if ESTACIONES[i][0] == "DCC":
                    out["hechas"] += 1
                    out["termino"] = env.now
                    out["ciclos"].append(env.now - t_ini)
                    if out["hechas"] == total:
                        fin.succeed()
                    yield from bandas[i].recorrer()   # regreso a Contenedores vacíos (0.5 min)
                    break
                yield from bandas[i].recorrer()
                i += 1

    for c in range(n_contenedores):
        env.process(contenedor(c))
    env.run(until=fin)
    for e in est:
        e._acum()
    return out, est


# ------------------------------------------------------------------ experimentos
def correr(n_rep, n_cont=10, semilla0=1000, detalle=False):
    term, filas = [], []
    for r in range(n_rep):
        out, est = replica(semilla0 + r, n_cont)
        T = out["termino"]
        term.append(T)
        if detalle:
            filas.append({e.nombre: (e.area_n / T, np.mean(e.permanencias),
                                     e.area_ocup / (e.proceso * T)) for e in est}
                         | {"_ciclo": np.mean(out["ciclos"])})
    return np.array(term), filas


if __name__ == "__main__":
    from scipy import stats
    real = np.array([949.1, 1043.8, 1040, 1125.2, 1159.3, 1115.3, 1081.7, 1135.2, 1021.4, 1108.3,
                     1047.7, 937.4, 1144.3, 1026.8, 1137, 1095.9, 1122.5, 1160.2, 921.5, 1030.7,
                     1087.5, 1088.4, 1094.2, 1154.6, 1131.8, 1213.6, 1169.4, 1313.9, 1032.7,
                     1083.5, 1009.9, 1022.3, 1005.3, 1030.2, 1016.4])
    piloto, _ = correr(20, semilla0=1)
    s = piloto.std(ddof=1)
    n_req = math.ceil((stats.t.ppf(0.975, 19) * s / 20) ** 2)
    print(f"Piloto 20 réplicas: media {piloto.mean():.1f}, s {s:.1f} -> n requerido = {n_req}")
    sim, det = correr(n_req, semilla0=5000, detalle=True)
    print(f"Corridas n={n_req}: media {sim.mean():.1f}, s {sim.std(ddof=1):.1f}")
    print(f"Real: media {real.mean():.1f}, s {real.std(ddof=1):.1f}")
    F = sim.var(ddof=1) / real.var(ddof=1)
    pF = 2 * min(stats.f.cdf(F, len(sim) - 1, 34), 1 - stats.f.cdf(F, len(sim) - 1, 34))
    t, pt = stats.ttest_ind(sim, real, equal_var=pF > 0.05)
    print(f"F = {F:.3f} (p {pF:.3f}) | t = {t:.3f} (p {pt:.3f})")
    print("\nEstación: inventario promedio | permanencia promedio | utilización")
    for nom in [e[0] for e in ESTACIONES]:
        v = np.array([f[nom] for f in det])
        print(f"  {nom:4s} {v[:,0].mean():6.2f} {v[:,1].mean():7.2f} {100*v[:,2].mean():6.1f}%")
    print(f"Ciclo promedio de un contenedor: {np.mean([f['_ciclo'] for f in det]):.1f} min")
    print("\nContenedores vs hora de terminación (30 réplicas c/u)")
    for nc in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30]:
        tt, _ = correr(30, nc, semilla0=9000)
        print(f"  {nc:3d}  {tt.mean():8.1f}")
