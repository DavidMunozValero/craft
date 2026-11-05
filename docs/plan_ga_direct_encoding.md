# Plan: GA con codificación directa (máscara + deltas) para maximización de revenue

Estado: viable y validado. Datos de entrada: los generados por el cuaderno
`0.0-Synthetic_Data_Generator.ipynb`. Alcance: objetivo único (revenue del IM),
algoritmo único (GA). Filosofía: simple y elegante — reutilizar al máximo la
capa compartida existente, sin tocar los backends actuales (gsa/mealpy).

> **Formulación formal** (función objetivo, restricciones, notación del paper
> y conexión ecuación ↔ código, con diagramas): ver
> `docs/approach_ga_direct_encoding.md`.

## 1. Motivación

El paper de referencia (`docs/papers/TimetablingProblemRailway`) concluye que
el GA es el mejor de los 10 metaheurísticos evaluados. En la formulación
(Ec. 1), `B_i` (servicio planificado) es variable de decisión binaria; en la
implementación actual no lo es (la fija la heurística greedy) y, además, la
matriz de conflictos se calcula solo con los horarios solicitados y no se
actualiza durante la búsqueda: desplazar tiempos no puede resolver conflictos.
La nueva codificación pone `B_i` y los deltas en el individuo y evalúa
conflictos sobre el horario decodificado.

## 2. Entrada de datos (cuaderno 0.0)

El cuaderno genera, con `seed=42`, dos archivos en `data/synthetic/`
(gitignored, regenerables):

| Archivo | Carga | Contenido |
|---|---|---|
| `timetable.yaml` | `robin.supply.entities.Supply.from_yaml` | 50 servicios Madrid→Barcelona (1 día, sentido outbound, 4 patrones), formato robin |
| `revenue.yaml` | `yaml.safe_load(...)["services"]` | `canon`, `ru`, `k`, `dt_max_penalty`, `tt_max_penalty`, `importance` por servicio — contrato exacto de `RevenueCalculator` |

- Horarios absolutos en minutos desde medianoche: `craft.scheduling.get_schedule_from_supply`
  los reconstruye directamente del supply.
- `safe_headway` (10) se lee de `timetable.yaml → metadata`; `im_mod_margin`
  (60) de `revenue.yaml → metadata.parameters` (con defaults por si faltan).
- La instancia ya contiene ~20 pares de salidas en conflicto: material de
  trabajo realista para la resolución dinámica de conflictos.

## 3. Codificación

Genoma mixto, minutos enteros, un bloque por servicio:

```
[b_i, δ_i^1, δ_i^2, ..., δ_i^{n_i-1}]
```

- `b_i ∈ {0,1}`: planificar o no el servicio (decisión explícita del GA).
- `δ_i^j ∈ ℤ`: delta respecto de la salida solicitada por el RU (0 = sin cambio);
  un gen por estación con salida; la llegada a destino se deriva
  (última salida + tiempo de marcha original), igual que en el código actual.

### Decodificación (arrastre acumulado con clamp)

```
cum_i^1 = δ_i^1                                  (δ_1 ∈ [−60, +60] en directos;
                                                  δ_1 ∈ [−60, min(60, Δ_max)] con paradas intermedias)
cum_i^j = min(cum_i^{j−1} + δ_i^j, Δ_max, cum_i^{j−1} + Δ_max − σ_i^j)   (δ_j ∈ [0, Δ_max])
t̃_i^j   = dt_i^j + cum_i^j
arrival_final_i = t̃_i^{n−1} + τ_i
```

El clamp replica los upper bounds actuales (`_calculate_boundaries`: margen de
dwell Δ_max = `max_stop_time`) y garantiza por construcción:

- tiempos de marcha y parada nunca se acortan (equivale a `δ_j ≥ 0`),
- hora de salida dentro del margen del IM en origen,
- extensiones de dwell dentro del máximo admisible.

Hallazgo de la implementación: en servicios con paradas intermedias, retrasar
el origen más allá de Δ_max es infactible (la llegada intermedia retrasada
choca con el cap de dwell), de ahí la cota superior del delta de origen según
el patrón (`_origin_delta_upper`). El propio clamp preserva la monotonía de
los acumulados ante genes fuera de bounds.

Ejemplo (de partida): referencias 8:00, 9:00, 10:00 y genoma `[1, 5, 0, 1]` →
acumulados (5, 5, 6) → **8:05, 9:05, 10:06**. Test unitario obligatorio.

Única restricción no estructural: **conflictos**, gestionados por penalización
pura en el fitness (sin heurística en el bucle).

## 4. Fitness

```
fitness = Σ_{i: b_i=1} revenue_i(horario decodificado)
          − λ · Σ_{(i,j) conflictivos con b_i=b_j=1} min(revenue_i, revenue_j)
```

- Revenue: `RevenueCalculator` sin cambios (revenue.yaml alimenta el contrato).
- Conflictos: **dinámicos**, recomputados sobre el horario decodificado con las
  condiciones cerradas del paper (gap de salida/llegada con signo opuesto o
  ≤ 2ω en cada arista del corredor compartida en el mismo sentido),
  interpolando linealmente con la distancia las estaciones de paso — la misma
  semántica de `robin.ServiceScheduler`. Checker propio vectorizado
  (`CorridorConflicts`), **validado 1:1 contra robin** en la instancia
  sintética (103/103 pares idénticos; el cuaderno 0.0 solo contaba conflictos
  en la estación origen). Instancia de 50 servicios → 1225 parejas, coste
  trivial (1.4 ms por evaluación completa).
- `min(rev_i, rev_j)` aproxima el coste real del conflicto (uno de los dos debe
  caerse). **`λ = 2` por defecto**: con `λ = 1` resolver un conflicto aislado
  sería indiferente (planificar ambos con la penalización rinde exactamente lo
  mismo que quedarse solo con el más caro), así que el GA no tendría presión
  para resolverlo; con `λ = 2` todo conflicto es estrictamente mejorable.
- El mejor individuo final se reporta con conflictos residuales (si los hay) y,
  como diagnóstico, el revenue que daría la heurística greedy sobre su conjunto
  b=1 (fuera del bucle).

## 5. Diseño

Dos módulos nuevos y un script; nada más:

```
src/craft/ga/
  timetabling.py   GATimetabling (clase pura: recibe el Supply y el revenue ya
                   cargados): estructura de bloques (slices por servicio),
                   decode (genoma → Solution con real=tiempos absolutos y
                   discrete=máscara), delta-bounds, fitness con conflictos
                   dinámicos; CorridorConflicts (checker por corredor)
  algorithm.py     GA: inicialización (b=1, δ=0+ruido), torneo (pop_size ≥ 10),
                   cruce por bloques de servicio, mutación (bit-flip en b;
                   gaussiana entera y reset-a-0 en δ), elitismo, convergencia
  __init__.py
scripts/run_ga.py  CLI: --timetable --revenue --seed --epoch --pop-size; carga
                   los YAML (incluidos los metadatos safe_headway e
                   im_mod_margin) y guarda convergence CSV y resumen del
                   mejor horario
```

Reutilización sin cambios: `craft.common` (Solution/Boundaries/ConflictMatrix),
`craft.revenue`, `craft.scheduling`. El backend no toca `runner.py`, `gsa/` ni
`mealpy/`; export lazy `craft.GATimetabling` en `craft/__init__.py`.
`data/` es gitignored: los resultados van a `data/results/`, la entrada se
regenera con el cuaderno.

## 6. Fases

| Fase | Contenido | Hecho cuando |
|---|---|---|
| 0 | Instancia base: cuaderno 0.0 → `data/synthetic/{timetable,revenue}.yaml` | ✅ 50 servicios, 103 pares en conflicto (semántica robin) |
| 1 | `timetabling.py`: decode + bounds + fitness | ✅ Ejemplo 8:05/9:05/10:06; property tests; checker validado 1:1 contra robin (103/103) |
| 2 | `algorithm.py`: GA completo | ✅ Determinista con seed; convergencia monótona (elitismo); history DataFrame; cuaderno `6.0-GA_Direct_Encoding.ipynb` para revisión visual |
| 3 | `scripts/run_ga.py` + export/docs | ✅ CLI e2e (convergencia, mejor horario y filmstrips en `data/results/`); exports perezosos `craft.GA`/`craft.GATimetabling`; docs (`optimizers.md`, `architecture.md`, `AGENTS.md`); visualización de individuos en `craft/ga/plotting.py` |
| 3b | Revisión y refactor: contrato `Problem`/`DiagnosableProblem` (`craft/ga/base.py`) para solvers intercambiables; `load_problem` + reporting (`describe_genome`, `schedule_frame`, `greedy_baseline`, `decoded_schedule`, horarios solicitados congelados) en el problema; rediseño de plotting (color por servicio, desvíos resaltados, `plot_timetable`/`plot_evolution`); cuaderno esbelto sobre helpers de `src` | ✅ cuaderno re-ejecutado y verificado visualmente |
| 4 | Experimento y comparación | GA vs mealpy GA/GSA en la misma instancia: revenue, n_scheduled, convergencia, tiempo |

Resultado destacado (fase 2): en la instancia sintética, el GA (pop 40,
100 generaciones, seed 42) alcanza un horario **sin conflictos** con revenue
**741.29**, superando a la heurística del paper sobre los horarios solicitados
(17 servicios, **729.53**). El cuaderno 6.0 lo reproduce con pop 50 × 150
generaciones (**745.29**) y permite inspeccionar individuos, decodificación,
convergencia y el diagrama espacio-tiempo final.

Criterio de éxito: revenue ≥ baselines en instancias con conflictos, con
menos o iguales servicios descartados, y desplazamientos que demuestren
resolución dinámica de conflictos (servicios salvados por el arrastre).

## 7. Riesgos y notas

- **Semántica del gen final**: se asume n−1 genes de salida por servicio y
  llegada a destino derivada (lectura consistente con el ejemplo de partida).
  Si se quiere delta explícito de llegada a destino, es un cambio acotado.
- **Penalización pura**: el GA debe aprender a evitar conflictos; si convergiera
  lento, el primer ajuste es subir λ (parámetro ya expuesto). No se añade
  reparación.
- El cuaderno 0.0 es hoy la única fuente de datos: cualquier cambio de CONFIG
  (n_services, seed, direcciones) regenera la instancia de forma reproducible.
