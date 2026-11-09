# Plan del paper 3 — Batería final: formulación bi-objetivo con codificación directa

Documento de referencia del tercer trabajo: tesis, **decisiones congeladas**,
diseño experimental final, metodología de análisis, **trabajo realizado** y
**trabajo pendiente**. Sustituye al anterior plan de publicación (brechas
G1–G7 y detalle P0–P7: `git log --follow -- docs/05_paper3_plan.md`).

> Documentación relacionada: formulación del encoding directo
> (`04_direct_encoding.md`), contratos y backends (`03_optimizers.md`),
> arquitectura (`01_architecture.md`).

## 0. Línea de publicación

| # | Trabajo | Codificación | Objetivo | Fuentes |
|---|---|---|---|---|
| 1 | *An approach to the timetabling problem in deregulated railway markets based on metaheuristic algorithms* | Salidas reales + máscara heurística (dos etapas) | Revenue del IM | `docs/papers/TimetablingProblemRailway/` |
| 2 | *Towards Equitable Rail Service Allocation Through Fairness-Oriented Timetabling in Liberalized Markets* | La del paper 1 | Revenue × equidad (multiplicativo, índices con capacidades) | `docs/papers/Fair_oriented_Timetabling/` |
| 3 | **Este trabajo** | **Directa** (máscara + deltas, factibilidad por construcción) | **Bi-objetivo** revenue ↔ equidad (frente de Pareto) | `docs/papers/BiObjective_Direct_Encoding/` (esqueleto) |

Revistas objetivo: *Integrated Computer-Aided Engineering* o *Computers &
Industrial Engineering*. Plan B: *Applied Soft Computing*, *Swarm and
Evolutionary Computation*.

## 1. Tesis y contribuciones

Narrativa: *de la optimización escalar con máscara heurística a la
formulación bi-objetivo con factibilidad por construcción, evaluada con una
ablación controlada en un marco de evaluación común.*

- **C1 — Codificación directa** (ablación: prueba 1 vs prueba 2, mismo
  tratamiento escalar rev×fair): máscara y deltas evolucionan juntos;
  factibilidad por construcción (`cumulative_shifts`); conflictos dinámicos
  recomputados sobre el horario decodificado (checker validado 1:1 contra
  robin, 103/103). La celda "vieja codificación + multiobjetivo" no existe
  por construcción: en la vieja codificación la máscara la fija una
  heurística (no es evolvable) y la equidad solo depende de la máscara, así
  que un segundo objetivo no sería optimizable. Este argumento justifica el
  diseño de 3 pruebas.
- **C2 — Tratamiento bi-objetivo** (ablación: prueba 2 vs prueba 3, misma
  codificación): el frente de Pareto real frente a un punto fijado a priori
  (producto multiplicativo heredado del paper 2).
- **C3 — Marco de evaluación común**: verificador dinámico de conflictos +
  filtro de operatividad + equidad operativa. Expone los defectos del
  esquema de dos etapas: conflictos nuevos tras optimizar salidas (su matriz
  de conflictos es estática), servicios operativamente inviables y equidad
  declarada que no sobrevive al filtro (0,93 declarado vs 0,40 operativo en
  la línea base histórica).
- **C4 — Paquete reproducible** (craft): contrato `Problem`, instancias
  generadas por cuaderno con semilla, CLIs, tests y batería reanudable.

## 2. Decisiones congeladas (2026-11-09)

| # | Decisión | Detalle |
|---|---|---|
| D1 | Fitness escalar de la prueba 2: `max(penalizado, 0) × fairness` | El revenue penalizado puede ser negativo (λ=2); el producto sin clamp invertiría el signo y premiaría horarios conflictivos e injustos. **Diferencia con el paper 2 a documentar en el manuscrito**: allí el fitness es `revenue crudo × fairness` (sin λ; los conflictos los eliminaba la máscara heurística sobre una matriz estática). Cada codificación conserva su paisaje de revenue nativo; el *tratamiento* (multiplicativo) es el heredado. |
| D2 | **Descongelar** la línea base; los GA mealpy parten de los valores publicados del paper 2 | `pc = 0.95`, `pm = 0.025` (papers 1–2). La tabla congelada del cuaderno 8.0 (pc 0.9 / pm 0.01, 5 semillas) queda como **referencia histórica**: ningún número del manuscrito saldrá de ella. Para los backends de codificación directa (sin valores publicados): `pc = 0.95` unificado con los mealpy y `pm = 0.05` (default del diseño del encoding, compartido por GA y NSGA-II al usar los mismos operadores). |
| D3 | **25 semillas** | Con n=5 el mínimo de Wilcoxon emparejado bilateral es p=0,0625 (>0,05): ninguna comparación sería significativa. n=25 es el rango 20–30 del protocolo estadístico. La lista de semillas se fija **antes de lanzar** (la batería reanuda por instancia-paso). Recomendación: conservar 7,21,42,63,84 (continuidad de apareamiento con la línea base histórica) + 20 adicionales. |
| D4 | Índice de equidad del objetivo: **Jain, α=25** | Justificación heredada del propio paper 2: Jain logra la menor inequidad en los escenarios balanced y unbalanced y la convergencia más estable; Gini y Atkinson quedan como métricas de *evaluación* (no de optimización). |
| D5 | Validación MO externa: **cross-check pymoo** en n50 | `scripts/run_pymoo.py` (grupo opcional `baselines`); criterio: C-metric sin dominación mutua masiva. La referencia exacta ε-restricción (Pyomo+SCIP) queda limitada a instancias de 25 servicios (intratabilidad documentada en 50/75, papel análogo a SCIP en el paper 1). |
| D6 | **GSA fuera** del estudio comparativo | Dominado por el GA en los papers 1–2 (y colapsa sin reparación). Se justifica en el manuscrito con esas referencias; no se re-evalúa. |
| D7 | La comparación con NSGA-II es **cuantitativa por semilla y por frente**, no por puntos | Nunca se promedian puntos del frente (ver §4.2). |

## 3. Diseño experimental final

### 3.1 Pruebas (núcleo de la batería)

| Prueba | Codificación | Tratamiento | Método en el repo | Rol | Estado |
|---|---|---|---|---|---|
| 1 | Vieja (real + máscara heurística) | Escalar `revenue × fairness` (Jain α=25) — el del paper 2 | `ga_mealpy_fair_jain` | Línea base del paper 2 | ✅ |
| 2 | **Directa** (máscara + deltas) | Escalar `max(penalizado, 0) × fairness` (Jain α=25) | `ga_direct_fair` (**nuevo**, T1) | **Ablación C1** (1 vs 2) | ❌ |
| 3 | **Directa** | Bi-objetivo `[penalizado, fairness]`, NSGA-II | `nsga2` | **Ablación C2** (2 vs 3) | ✅ |

Cada salto de prueba cambia exactamente una cosa: 1→2 la codificación,
2→3 el tratamiento de objetivos.

### 3.2 Anclas (filas de referencia, fuera de la ablación)

- **Greedy** (`greedy_baseline()`): determinista, una sola ejecución por
  instancia, coste nulo. Techo estructural del esquema de dos etapas sobre
  los horarios solicitados (729,53 en n50_r3).
- **GA-mealpy mono** (paper 1, `ga_mealpy`): *recomendado mantener* como
  fila (coste ~48 s/semilla en n50): es el portador de la evidencia C3 de
  conflictos post-optimización y el representante del paper 1 en las tablas.
  No participa en la ablación. [Decisión abierta menor, §7]

### 3.3 Instancias, presupuesto y configuración

- **Instancia núcleo**: `n50_r3` (`data/synthetic/instances/n50_r3/`,
  byte-idéntica a la raíz del cuaderno 0.0: 50 servicios Madrid→Barcelona,
  3 RUs, 103 pares en conflicto sobre los horarios solicitados).
- **Extensión de escalabilidad** (decisión abierta, §7): n25_r3 / n75_r3
  para la tabla 25/50/75 del manuscrito; los escenarios de cuota
  (r5_balanced/semi/unbalanced) quedan como extensión posterior.
- **Presupuesto**: pop 40 × epoch 100 = 4000 evaluaciones para todas las
  metaheurísticas; λ = 2; ω = 10; δ = 60; Δ_max = 10.
- **Semillas**: 25 (D3).
- **Estimación de cómputo** (n50, ~48 s/semilla/run medida): 4 runs
  meta × 25 semillas ≈ 80 min/instancia en un hilo; cross-check pymoo aparte.

## 4. Metodología de análisis

### 4.1 GA escalar vs GA escalar (ablación C1: prueba 1 vs 2)

Métricas del evaluador común (`craft.ga.evaluation`): revenue,
revenue penalizado, servicios operativos, conflictos residuales, Jain
operativo, Gini/Atkinson, inequidad %. Comparación emparejada por semilla:
Wilcoxon signed-rank + delta de Cliff (bandas Romano) + corrección de Holm
por familia (instancia, métrica) — `craft.stats`, `scripts/run_stats.py`.
Además: curvas de convergencia por generación en el plano (revenue, fairness)
(gracias al logging de T2) y curvas de fitness escalar.

### 4.2 NSGA-II: por qué no se promedian los puntos del frente

Las soluciones del frente son puntos **no comparables entre sí**: uno lleva
el revenue al máximo con equidad mínima y otro lo contrario. Promediar sus
coordenadas (o su desviación) no mide ni calidad ni diversidad — mezcla
soluciones de compromiso distintos. Por eso la comparación cuantitativa con
NSGA-II se hace **por semilla y a tres niveles**, cada uno con *un* número
por run (nunca un promedio sobre puntos del frente):

| Nivel | Qué se calcula | Comparación |
|---|---|---|
| **A — Rodilla (knee)** | Una solución representativa por run: el punto del frente más cercano a la esquina ideal tras normalización min-max (`knee_index`, ya implementado). Sus métricas del evaluador común → media ± sd **entre semillas**. | 1:1 con los GA escalares (que también producen una solución por semilla). Va a las tablas del manuscrito. |
| **B — Indicadores de frente** | Cada frente se resume en **un escalar por semilla**: hipervolumen normalizada con referencia nadir (sobre la unión de frentes), IGD contra el frente de referencia consolidado (unión de frentes ∪ puntos escalares ∪ ε), spread Δ. Media ± sd **de los indicadores** entre semillas. | Calidad/diversidad del frente completo. Estándar del área (Deb; Zitzler). |
| **C — Puente de dominancia** | `c_metric(frente, puntos)` y % de puntos de las pruebas 1–2 (y del greedy) débilmente dominados por el frente, por semilla. | La respuesta cuantitativa a "¿merece la pena el tratamiento bi-objetivo?": **criterio de aceptación de la tesis** — el frente domina o cubre lo que los escalares muestrean con su único punto. |
| D — Cualitativo | Forma del frente, extremos, Marey de la rodilla y de los extremos, trayectorias (revenue, fairness) por generación frente al frente final. | Complemento visual, no método de comparación. |

### 4.3 Estadística

Wilcoxon signed-rank emparejado por semilla; delta de Cliff con bandas de
magnitud (Romano); corrección de Holm por familia (instancia, métrica);
medias ± desviación. Con 25 semillas todas las comparaciones del núcleo son
demostrables a α=0,05 si el efecto es real.

### 4.4 Evidencia C3: equidad declarada vs operativa

Para la prueba 1 (dos etapas), reportar la equidad sobre la máscara
**declarada** por la heurística y sobre la máscara **operativa** (tras el
filtro de factibilidad del evaluador), junto con conflictos-en-solicitados
vs conflictos-en-optimizados. Requiere la extensión pequeña del evaluador
(T4).

## 5. Trabajo realizado (habilitador, todo testeado)

| Pieza | Dónde |
|---|---|
| Codificación directa + GA (factibilidad por construcción, conflictos dinámicos, checker validado 1:1 contra robin) | `craft/ga/` — formulación en `04_direct_encoding.md` |
| NSGA-II bi-objetivo (revenue, fairness) con operadores compartidos con el GA | `craft/ga/nsga2.py`, `craft/ga/objectives.py` |
| Evaluador común (verificador dinámico + filtro de operatividad + Jain/Gini/Atkinson con capacidades + inequidad %) | `craft/ga/evaluation.py` |
| Indicadores MO (HV nadir normalizada, IGD, spread, C-metric) | `craft/ga/indicators.py` |
| Estadística (Wilcoxon, Cliff+Romano, Holm) | `craft/stats.py`, `scripts/run_stats.py` |
| Runner comparativo método × semilla con artefactos estructurados | `scripts/run_comparison.py` |
| Batería reanudable E1–E4 (subprocesos por instancia, saltos con artefacto) | `scripts/run_battery.py` |
| Familia de instancias 25/50/75 × {r3, r5×3} con manifiesto | cuaderno 0.0 §8 → `data/synthetic/instances/` |
| Referencia exacta ε-restricción (solo 25 servicios) | `scripts/run_epsilon.py` (opcional `baselines`) |
| Cross-check pymoo | `scripts/run_pymoo.py` (opcional `baselines`) |
| Determinismo de la heurística (desempate por revenue + id) | `MealpyTimetabling.get_heuristic_schedule` (P2) |
| Esqueleto del manuscrito | `docs/papers/BiObjective_Direct_Encoding/main.tex` |

Línea base histórica (5 semillas, config antigua, **no** fuente de números
del manuscrito): GA directo 724,68 ± 27,30 sin conflictos; NSGA-II knee
723,79 ± 42,74 con Jain operativo 0,85 ± 0,16; método del paper 2 con
equidad declarada 0,93 vs operativa 0,40.

## 6. Trabajo pendiente (backlog)

| # | Tarea | Criterio de hecho | Esfuerzo |
|---|---|---|---|
| T1 | **Fitness escalar rev×fair para el encoding directo**: builder en `craft/ga/objectives.py` (adapter sobre el contrato `Problem` con `fitness = max(penalizado, 0) × fairness(Jain α=25)`) + método `ga_direct_fair` en `run_comparison.py`/`run_battery.py` + tests | Tests: signo (penalizado negativo ⇒ fitness 0, nunca premiado), máscara vacía ⇒ (0,·), determinismo por semilla; smoke run e2e | ~medio día |
| T2 | **Logging por generación de (penalizado, fairness)** del mejor individuo en los dos GA escalares (NSGA-II ya lo tiene vía `pareto_history`) | Columnas nuevas (o CSV adicional) en el historial; figura de trayectoria en el plano objetivo | horas |
| T3 | **Config de batería congelada**: overrides pc=0.95 / pm=0.025 (mealpy), pc=0.95 (GA directo y NSGA-II), lista de 25 semillas fijada; docstring/README de los scripts con la config del paper | Un comando lanza la batería completa del núcleo con la config D2/D3 sin flags extra | horas |
| T4 | **Equidad declarada vs operativa** en el evaluador (E6): máscara declarada (heurística pre-filtro) vs operativa para los métodos de dos etapas | Columnas `fairness_jain_declared` / `planned_declared` en `runs.csv`; figura solicitados-vs-optimizados | horas |
| T5 | **Análisis de la batería**: consolidación del frente de referencia (unión ∪ escalares ∪ ε), C-metric frente-vs-puntos (nivel C), curvas HV/IGD por generación, stats Wilcoxon/Cliff/Holm | `stats.csv`, `c_metric.csv` y figuras E7 por instancia | ~1 día |
| T6 | *(Abierta, §7)* Extensión de instancias n25/n75 (+ escenarios de cuota) para escalabilidad | Tabla 25/50/75 con calidad y tiempo | batería extra |
| T7 | **Redacción**: actualizar §Diseño experimental del manuscrito al alcance real de esta batería (hoy promete "12 instancias, 8 métodos, 25 semillas"); figuras E7 (money-figure, escalabilidad, reparto por RU, Marey); tabla resumen con significancia | Manuscrito coherente con §3–§4 de este plan | ~1 semana |

**Orden**: T1 → T3 → smoke test (5 semillas en `--output-dir` temporal) →
lanzamiento 25 semillas → T2/T4 (en paralelo con la batería) → T5 → T7.

## 7. Decisiones abiertas

| # | Decisión | Recomendación |
|---|---|---|
| A1 | Extensión a n25/n75 (T6) para la tabla de escalabilidad y las afirmaciones de la sección Resultados del esqueleto | Sí, tras el núcleo n50: la batería es reanudable y `--methods/--instances` son flags; fijar las mismas 25 semillas |
| A2 | Mantener `ga_mealpy` mono (paper 1) como fila ancla de C3 | Sí (~48 s/semilla; sin él la tabla pierde al paper 1) |
| A3 | Ablaciones E5 posteriores (λ ∈ {1,2,5,10}, ω×δ con ANOVA/re-gímenes de señalización, M=1 vs M=2, dominación de Deb en el extremo de equidad) | Post-batería, según lo que pidan revisores/espacio; el núcleo no depende de ellas |

## 8. Riesgos

- **C3 es una afirmación fuerte sobre los métodos previos** (mismos
  autores): evidencia irrefutable (T4, checker validado) y tono no defensivo.
- **El extremo de equidad del frente puede arrastrar conflictos** (la
  equidad solo depende de la máscara): si aparece, documentarlo como
  limitación o resolverlo con la variante de dominación de Deb (A3).
- **ε-restricción no escala** más allá de 25 servicios: documentarlo como
  límite del exacto, no del método propuesto.
- **Reproducibilidad**: determinismo P2 resuelto; prometer artefactos solo
  con la lista de semillas congelada (D3).
- **Sobreafirmación ERTMS** (§9): solo "condición de separación de origen
  señalización, como surrogate macroscópico"; nunca "modelamos ETCS" ni
  vocabulario de certificación (EN 50129).

## 9. Encuadre señalización/protección (ERTMS) — solo narrativo

| Elemento del modelo | Lectura ERTMS / señalización |
|---|---|
| Conflicto `gap ≤ 2ω` y cruce por arista compartida en el mismo sentido | Separación de la capa de protección: *blocking times* (bloque fijo), curvas de frenado (ETCS L2), cantón móvil (ETCS L3) |
| `safe_headway ω` (umbral `2ω`) | Proxy macroscópico del régimen de protección; versión macroscópica de la tradición *blocking time* / UIC 406 |
| Factibilidad por construcción + checker dinámico | Paths "operables" que la protección concedería, frente al horario "de papel" |
| C3 (conflictos post-optimización) | Métodos optimistas que proponen paths que la señalización rechazaría |
| Máscara `B_i` + deltas + margen IM | Asignación de *train paths*: el RU solicita, el IM concede/ajusta (TTR) |
| Bi-objetivo revenue IM ↔ equidad entre RUs | Dilema del IM que formalizan la Directiva 2012/34/UE y el reglamento de capacidad TTR |

Bibliografía dirigida pendiente de verificación antes de citar: número y
estado del reglamento TTR en EUR-Lex; CSM ADA (no confirmado — citar solo
UIC 406 + *blocking time* si no se confirma); cifra de capacidad ~40% desde
el factsheet oficial ERTMS/ERA (no literatura secundaria). Ítems: *blocking
time theory*/UIC 406; Bešinović & Goverde (2021); Van Aken et al. (TR-B
2017); estudios *moving block* (Knutsen et al., IEEE 2023; JRTPM 2025);
Directiva 2012/34/UE; TTR.
