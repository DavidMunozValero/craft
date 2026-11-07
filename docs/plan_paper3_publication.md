# Plan de publicación: tercer paper (formulación bi-objetivo con codificación directa)

Objetivo: evaluar y planificar qué falta para trasladar el trabajo actual
(codificación directa + NSGA-II bi-objetivo revenue/equidad) a una nueva
publicación de primer nivel en la línea de planificación de horarios, tras
*An approach to the timetabling problem in deregulated railway markets based
on metaheuristic algorithms* (`docs/papers/TimetablingProblemRailway/`) y
*Towards Equitable Rail Service Allocation Through Fairness-Oriented
Timetabling in Liberalized Markets* (`docs/papers/Fair_oriented_Timetabling/`).

Revistas objetivo: **Integrated Computer-Aided Engineering** o
**Computers & Industrial Engineering** (ambas publicaron trabajos de la línea;
encaje directo). Plan B: *Applied Soft Computing*, *Swarm and Evolutionary
Computation* (más centradas en la contribución metaheurística/MO).

> Contexto técnico del encoding: `docs/approach_ga_direct_encoding.md`.
> Plan original del GA directo: `docs/plan_ga_direct_encoding.md`.

## 1. Estado actual (línea base congelada en el cuaderno 8.0)

Experimento comparativo sobre la instancia común del cuaderno 0.0 (50
servicios Madrid→Barcelona, 3 RUs, 103 pares en conflicto), mismo presupuesto
para todas las metaheurísticas (pop 40 × 100 gen = 4000 evaluaciones), 5
semillas (7/21/42/63/84), evaluador común (verificador dinámico de conflictos
validado 1:1 contra robin + filtro de operatividad + 3 índices de equidad con
capacidades + inequidad del paper fair-oriented):

| Método | Revenue crudo | Penalizado (λ=2) | Operativos | Conflictos | Jain operativo |
|---|---|---|---|---|---|
| Greedy (paper 1) | 729,53 | 729,53 | 17,0 | 0 | 0,33 |
| GA-mealpy mono (paper 1) | 583,05 ± 39,54 | 395,44 ± 117,79 | 15,4 ± 1,1 | 2,6 ± 1,8 | 0,37 ± 0,07 |
| GA-mealpy rev×Jain(25) (paper 2) | 456,30 ± 67,00 | 339,61 ± 24,95 | 12,2 ± 1,9 | 1,6 ± 1,1 | 0,40 ± 0,09 |
| GSA (craft, sin reparación) | 71,26 ± 25,07 | 57,45 ± 18,24 | 2,0 ± 0,7 | 0,2 ± 0,5 | 0,33 |
| **GA directo** | **724,68 ± 27,30** | **724,68 ± 27,30** | **17,6 ± 0,9** | **0** | 0,44 ± 0,15 |
| **NSGA-II bi-objetivo (knee)** | 723,79 ± 42,74 | 627,34 ± 35,07 | 17,6 ± 1,1 | 1,2 ± 0,8 | **0,85 ± 0,16** |

Hallazgos clave (sección de discusión del cuaderno 8.0):

1. El esquema de dos etapas publicado acumula dos fugas estructurales:
   conflictos nuevos tras optimizar salidas (su matriz de conflictos es
   estática, de los horarios solicitados) y servicios que quedan operativamente
   inviables (la cota secuencial de salidas intermedias no se impone al muestrear).
2. La equidad que persigue el objetivo multiplicativo del paper 2 se mide sobre
   la máscara heurística y **no sobrevive al filtro de operatividad**
   (Jain operativo 0,40 frente a ~0,93 declarado en la mejor semilla).
3. El GA directo termina sin conflictos en 5/5 semillas y supera a los métodos
   de dos etapas en la métrica estricta en 5/5 parejas (Wilcoxon, p = 0,0625,
   el mínimo alcanzable con n = 5).
4. NSGA-II traza el frente revenue–equidad completo (8–22 puntos por semilla;
   equidad extrema Jain ≈ 0,99); su mejor punto estricto queda bajo el GA
   directo en 5/5 semillas sin significancia (p = 0,4375): el coste del segundo
   objetivo queda visible y cuantificado.
5. El backend GSA de craft **colapsa sin reparación** (2,0 ± 0,7 trenes
   operativos). Diagnóstico empírico: sus posiciones violan la cota secuencial
   (p. ej. salida intermedia 511,3 con cota inferior 545) y su propio objetivo
   descarta los servicios inviables; su convergencia interna es plana (~224).
   GSA_M lo resolvía con `repair_solution=True` (clip secuencial con cotas
   dinámicas) y población sembrada con el horario solicitado; el port a craft
   perdió ambos mecanismos. La reparación propuesta reutiliza `cumulative_shifts`
   (decodificador del encoding directo) como operador de reparación.

## 2. Tesis y contribuciones propuestas

Narrativa: *de la optimización escalar con máscara heurística a la
formulación bi-objetivo con factibilidad por construcción, evaluada en un
benchmark común.*

- **C1 — Codificación directa**: máscara y deltas evolucionan juntos;
  factibilidad por construcción (`cumulative_shifts`); conflictos dinámicos
  recomputados sobre el horario decodificado (checker validado 1:1 contra
  robin, 103/103). Ablation que la aísla: GA-mealpy vs GA directo (mismo
  presupuesto).
- **C2 — Reformulación bi-objetivo** (revenue del IM ↔ equidad entre RUs) con
  NSGA-II propio sobre el mismo encoding: el frente de Pareto real en lugar de
  un punto fijado a priori (peso aditivo en el runner, producto multiplicativo
  en el paper 2). Ablation que la aísla: NSGA-II con M=1 (degenera en GA
  elitista; ya demostrado en el cuaderno 7.0) vs M=2.
- **C3 — Marco de evaluación común** que expone defectos estructurales del
  enfoque de dos etapas (conflictos post-optimización, inviabilidad operativa,
  equidad no operativa). Requiere la figura conflictos-en-solicitados vs
  conflictos-en-optimizados y la validación del verificador. Lectura
  señalización/protección (ver 3b): el verificador dinámico no solo cuenta
  conflictos — verifica factibilidad contra la capa de protección sobre el
  horario optimizado: "horario operable" frente a "horario de papel".
- **C4 — Paquete reproducible** (craft): contrato `Problem`, instancia
  generada por cuaderno con semilla, CLIs y tests.

La métrica de equidad es **heredada** del paper 2 (Jain/Gini/Atkinson con
capacidades y α calibrado); lo nuevo es el *tratamiento* multi-objetivo, no la
métrica. Debe declararse explícitamente.

## 3. Evaluación de publicabilidad (brechas)

Veredicto: el esqueleto conceptual (C1+C2+C3) tiene entidad para revista de
primer nivel, pero el estado actual (1 instancia × 5 semillas, sin baselines
MO estándar ni indicadores completos) es de congreso. Brechas:

| # | Brecha | Qué la cierra |
|---|---|---|
| G1 | Una sola instancia (50 serv., 3 RUs) frente a la escalabilidad 25/50/75 y los 3 marcos de capacidad de los papers previos | E1 |
| G2 | Sin baselines MO estándar ni referencia exacta | P4 + E2 (pymoo, ε-restricción SCIP) |
| G3 | Indicadores MO incompletos (solo HV con ref (0,0)) | P3 + E4 (HV nadir, IGD, spread, C-metric) |
| G4 | Sin ablations formales (encoding y M=1/M=2 están solo demostrados informalmente) | E5 |
| G5 | 5 semillas, sin corrección por comparaciones múltiples ni tamaños de efecto | E3 |
| G6 | Baseline GSA inválido (sin reparación) — atacable por revisores | P0 |
| G7 | Posicionamiento frente a literatura MO en timetabling (NSGA-II con objetivos de pasajeros/energía existe; el nicho = revenue IM ↔ equidad de operadores con asignación de servicios y encoding directo) | Redacción + búsqueda bibliográfica dirigida (encuadre señalización/protección y bibliografía: sección 3b) |

### 3b — Encuadre señalización/protección (ERTMS) y bibliografía dirigida (G7)

El checker de conflictos admite una lectura natural en términos de la capa de
protección ferroviaria; se incorpora como encuadre narrativo (sin código nuevo
ni experimentos añadidos):

| Elemento del modelo | Lectura ERTMS / señalización |
|---|---|
| Condición de conflicto `gap ≤ 2ω` y cruce por arista compartida en el mismo sentido | Separación que impone la capa de protección: escalera de *blocking times* en bloque fijo, curvas de frenado en ETCS L2, cantón móvil en ETCS L3 (*moving block*) |
| `safe_headway ω` (umbral `2ω`) | Proxy macroscópico del régimen de protección; versión macroscópica de la tradición *blocking time* / UIC 406 |
| Factibilidad por construcción + checker dinámico | Paths "operables" que la protección concedería, frente al horario de papel que el interlocking invalidaría |
| C3 (conflictos post-optimización del flujo de dos etapas) | Métodos optimistas que proponen paths que la señalización rechazaría; el checker dinámico como gemelo macroscópico de esa capa |
| Máscara `B_i` + deltas + margen IM | Asignación de *train paths*: el RU solicita, el IM concede/ajusta (*capacity-sensitive path adjustment* del TTR) |
| Bi-objetivo revenue IM ↔ equidad entre RUs | Dilema real del IM que formalizan la Directiva 2012/34/UE (asignación de capacidad, no discriminación, cánones) y el nuevo reglamento de capacidad basado en TTR |

Ubicación en el manuscrito:

| Sección | Contenido |
|---|---|
| Intro / motivación | Párrafo regulatorio: en mercado liberalizado el IM construye horarios *capacity-sensitive* que la señalización debe poder garantizar en explotación; marco TTR / 2012/34 |
| Modelo / restricciones | Remark: `2ω` es una restricción de origen señalización; ω parametriza el régimen de protección |
| Discusión | Certificabilidad de paths (lectura ERTMS de C3) + límites del surrogate macroscópico |
| Future work | Headway por pareja de trenes `ω_ij = f(τ_i, τ_j)` en `CorridorConflicts` — la esencia del *moving block*; cambio acotado |
| Related work | Ítems de la búsqueda dirigida (abajo) |

Búsqueda bibliográfica dirigida:

- *Blocking time theory* y método **UIC 406** (tradición macroscópica de
  capacidad, donde encaja el umbral `2ω` del checker).
- Review de timetabling: **Bešinović & Goverde** (*Review of railway
  timetabling developments: planning phases, goals and future directions*,
  2021); **Van Aken, Bešinović & Goverde** (TTAP, *Transportation Research
  Part B*, 2017).
- Capacidad **ERTMS/ETCS**: estudios de *moving block* (Knutsen et al.,
  IEEE 2023; JRTPM 2025); factsheet oficial ERTMS/ERA *Increasing
  infrastructure capacity* (~40% de ganancia de capacidad al sustituir
  bloque fijo).
- Marco regulatorio: **Directiva 2012/34/UE** y reglamento de capacidad
  basado en **TTR / Timetable Redesign** (acuerdo político, nov. 2025).

Verificaciones antes de citar: (1) número y estado definitivo del reglamento
TTR en EUR-Lex (aparentemente adoptado en 2026); (2) número del reglamento UE
de evaluación de capacidad (CSM ADA) — no confirmado en búsquedas, verificar
o citar solo UIC 406 + *blocking time*; (3) la cifra de capacidad (~40%)
citarla del factsheet ERTMS/ERA oficial, no de literatura secundaria.

## 4. Prerrequisitos de código

| # | Tarea | Criterio de hecho | Esfuerzo | Estado |
|---|---|---|---|---|
| P0 | Reparación del GSA: tras cada movimiento, deltas → `cumulative_shifts`; población inicial sembrada con el horario solicitado | GSA reparado ≥ heurística greedy en la instancia 50 y sin colapso operativo (≥ 15 trenes operativos) | ~1 día | ✅ `MealpyTimetabling.repair_departures` (primitive `cumulative_shifts` moved to `craft/scheduling`); wiring en `ExperimentRunner._optimize_gsa` y `scripts/run_comparison.py`. Criterio cumplido: fitness 729,53 con 17 trenes en la CLI; en el cuaderno 8.0 corregido, GSA 728,82 ± 0,01 sin conflictos y 17,0 trenes (antes: 2,0 ± 0,7 operativos y 57,45) |
| P1 | Runner batch `scripts/run_comparison.py`: rejilla método × instancia × semilla → artefactos estructurados (`data/results/benchmark/`) | Un comando lanza la rejilla completa y produce los CSV de E7 | 1–2 días | ✅ Métodos `greedy/ga_mealpy/ga_mealpy_fair/gsa/ga_direct/nsga2` × semillas sobre una instancia; artefactos `runs.csv`, `summary.csv`, convergencias, frentes, horarios por run; evaluador común extraído a `craft/ga/evaluation.py` (`SolutionEvaluator`, `inequity_percent`) y `knee_index` movida a `craft/ga/nsga2.py`. Multi-instancia se resuelve invocando por instancia (P5 añadirá el generador) |
| P2 | Determinismo de `get_heuristic_schedule`: desempate determinista (el `set` de conflictivos itera con hash aleatorizado → no reproducible entre procesos) | Dos procesos con PYTHONHASHSEED distinto producen resultados idénticos | horas | ✅ Desempate por revenue descendente + id ascendente (`sorted(key=(-rev, id))` + `next(iter(...))`); test con empate forzado (`tests/test_mealpy_timetabling.py`) |
| P3 | Indicadores MO con tests: HV normalizada con referencia nadir, IGD, spread Δ, C-metric | Tests de casos conocidos + uso desde P1 | ~1 día | Pendiente |
| P4 | Cross-check pymoo NSGA-II sobre el contrato `Problem`; modelo ε-restricción Pyomo+SCIP para frentes de referencia (instancia 25) | Frente pymoo ≈ frente propio (C-metric sin dominación mutua masiva); puntos ε-restricción en el plano revenue–equidad | ~2 días | Pendiente |
| P5 | Cuaderno 0.0 ampliado: instancias con cuotas de RU (5 RUs; balanced 20%×5, semi 30/25/20/15/10, unbalanced 55/25/10/5/5) y tamaños 25/50/75, semilla fija | Instancias versionadas en `data/synthetic/` con manifiesto | ~1 día | Pendiente |

## 5. Batería experimental

**E1 — Instancias**: 25/50/75 servicios × {3 RUs actual, 5 RUs balanced, 5 RUs
semi, 5 RUs unbalanced} (regenerables por P5, semilla fija declarada).

**E2 — Métodos por instancia**: greedy · GA-mealpy mono (paper 1) · GSA
reparado · GA directo · NSGA-II bi-objetivo · GA-mealpy rev×fair con
{Revenue, Jain(25), Gini(10), Atkinson(25)} (puntos del paper 2) · pymoo
NSGA-II (cross-check) · ε-restricción exacta (solo 25 servicios; en 50/75
documentar intratabilidad, replicando el papel de SCIP en el paper 1).

**E3 — Presupuesto y estadística**: mismo nº de evaluaciones para todas las
metaheurísticas; **20–30 semillas** por celda; Wilcoxon/Mann-Whitney con
corrección de Holm; tamaño de efecto (delta de Cliff); medias ± desviación.

**E4 — Indicadores MO**: HV con referencia nadir normalizada; IGD contra el
frente de referencia (unión de frentes ∪ puntos ε-restricción); spread Δ;
C-metric de NSGA-II frente a (i) barrido suma-ponderada y (ii) barrido
rev×fair con α ∈ {1, 5, 10, 25, 50}. Criterio de aceptación de la tesis: el
frente de NSGA-II (un run) domina o cubre lo que los escalares muestrean con
todas sus configuraciones (C-metric alta, HV front ≥ HV unión de puntos).

**E5 — Ablations**: encoding (GA-heurístico vs GA-directo, mismos operadores y
presupuesto); M=1 vs M=2 en NSGA-II; λ ∈ {1, 2, 5, 10} y dominación por
restricciones de Deb vs penalización (responde la limitación del extremo de
equidad); índice de equidad del objetivo 2 y sensibilidad de α; sensibilidad
ω ∈ {2.5, 5, 10} × δ ∈ {30, 45, 60} con ANOVA (réplica del análisis del paper 1);
el barrido de ω se interpreta además como análisis de **regímenes de
señalización** (ω=10 ≈ bloque fijo convencional, ω=5 ≈ ETCS L2, ω=2.5 ≈
ETCS L3/*moving block*, ver 3b): mismo cómputo, etiqueta nueva + párrafo de
discusión sobre cómo se mueve el frente revenue–equidad con el régimen de
protección.

**E6 — Robustez de C3**: figura conflictos-en-solicitados vs
conflictos-en-optimizados por método y instancia; distribución de servicios
inoperativos; equidad declarada vs operativa para el método del paper 2.

**E7 — Figuras y tablas del manuscrito**: money-figure por instancia (frente
vs puntos mono-objetivo); HV/IGD vs generación; tabla de escalabilidad
(calidad y tiempo vs nº servicios); asignación por RU por escenario; Marey del
knee y de los extremos; tabla resumen media ± desviación con significancia.

## 6. Riesgos

- **C3 es una afirmación fuerte sobre los métodos previos**: al ser los mismos
  autores, el mensaje "fortalecemos nuestra formulación" es natural, pero la
  evidencia debe ser irrefutable (verificador validado, E6) y el tono no
  defensivo.
- **ε-restricción puede no escalar** más allá de 25 servicios (como SCIP en el
  paper 1): documentarlo como límite del exacto, no del método propuesto.
- **El frente en el extremo de equidad arrastra conflictos** (la equidad solo
  depende de la máscara): si la variante con dominación por restricciones (E5)
  lo resuelve sin pérdida de HV, incluirlo como resultado; si no, como limitación.
- **Reproducibilidad**: P2 es condición necesaria para prometer artefactos
  reproducibles en el paper.
- **Sobreafirmación ERTMS/señalización** (ver 3b): el modelo es macroscópico a
  nivel corredor — sin cantones, curvas de frenado, RBC ni transiciones de
  nivel. La única afirmación permisible es "condición de separación de origen
  señalización, como surrogate macroscópico": nunca reclamar "modelamos ETCS"
  ni usar vocabulario de certificación de seguridad (EN 50129). La conexión
  entra como encuadre narrativo (3b) y relabel del análisis ω (E5), no como
  contribución; la extensión `ω_ij` queda en *future work*.

## 7. Orden de ejecución

1. ~~P0–P2 (correcciones de línea base) → re-ejecutar el cuaderno 8.0 y
   congelar~~. **Hecho**: cuaderno 8.0 re-ejecutado con GSA reparado y CaSP
   determinista; discusión actualizada. Resultado nuevo relevante: el GSA
   reparado alcanza 728,82 ± 0,01 sin conflictos — el techo estructural del
   esquema de dos etapas es la heurística greedy sobre los horarios
   solicitados (729,53) — y GA directo vs GSA deja de ser significativo
   (p = 0,8125). El diferencial del encoding directo ya no está en revenue
   estricto sino en (i) factibilidad sin parches externos, (ii) superar el
   techo greedy en nº de trenes (17,6 ± 0,9) y (iii) la extensión
   bi-objetivo.
2. P3–P5 (indicadores, baselines MO, generador de instancias).
3. E1–E4 (batería principal; cómputo estimado: horas–días).
4. E5–E6 (ablations y robustez).
5. Redacción (C1–C4, E7, limitaciones).
