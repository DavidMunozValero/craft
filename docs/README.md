# Documentación de craft

Los documentos de este directorio están **numerados en orden de lectura**
(`NN_nombre.md`). Convenciones:

- Los números son estables: un documento conservará su número aunque cambie
  su contenido; los documentos nuevos toman el siguiente número libre.
- No reutilizar números de documentos eliminados (su historial vive en git).
- Las fuentes LaTeX de los papers viven en `papers/` (sin numerar: no son
  documentación del marco, son entregables).

## Índice

| # | Documento | Idioma | Contenido |
|---|---|---|---|
| 01 | [01_architecture.md](01_architecture.md) | EN | Arquitectura del paquete: capas, módulos, flujo de datos de un experimento, configuración. |
| 02 | [02_problem_formulation.md](02_problem_formulation.md) | EN | Formulación del problema: variables de decisión, restricciones, modelo de revenue (canon y penalizaciones) e índices de equidad. |
| 03 | [03_optimizers.md](03_optimizers.md) | EN | Backends de optimización: GSA, mealpy, GA directo y NSGA-II; evaluador común, indicadores MO, baselines externos (pymoo, ε-restricción), batería y estadística; cómo añadir algoritmos. |
| 04 | [04_direct_encoding.md](04_direct_encoding.md) | ES | Formulación detallada del encoding directo: notación del paper 1 ↔ código, individuo y decodificación, función objetivo, restricciones, conflictos dinámicos, instancia de referencia y mapa del código. |
| 05 | [05_paper3_plan.md](05_paper3_plan.md) | ES | Plan de la batería final del tercer paper: decisiones congeladas, diseño experimental (3 pruebas + anclas), metodología de análisis, trabajo realizado y backlog. |

## Guía de lectura por rol

- **Nuevo en el repositorio**: 01 → 02 → 03 (marco técnico completo).
- **Trabajar en la codificación directa o los operadores genéticos**: 04
  (formulación ecuación a ecuación) con 01/03 de referencia.
- **Ejecutar o analizar la batería del paper 3**: 05 (decisiones congeladas
  D1–D7, diseño §3, análisis §4, backlog §6) + 03 para los scripts.
- **Agentes**: las convenciones del repo (comandos, estilo, robin como
  dependencia editable, checklist pre-commit) están en el `AGENTS.md` de la
  raíz; este índice es el mapa de la documentación.

## Papers

| Directorio | Trabajo |
|---|---|
| `papers/TimetablingProblemRailway/` | Paper 1: timetabling en mercados liberalizados con metaheurísticos (mealpy; salidas reales + máscara heurística). |
| `papers/Fair_oriented_Timetabling/` | Paper 2: timetabling orientado a equidad (revenue × fairness multiplicativo). |
| `papers/BiObjective_Direct_Encoding/` | Paper 3 (en curso): formulación bi-objetivo con codificación directa — esqueleto `main.tex`. |
