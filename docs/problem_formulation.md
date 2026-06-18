# Problem Formulation

CRAFT formulates the Infrastructure Manager's (IM) revenue-maximization
timetabling problem. The IM receives service requests from Railway
Undertakings (RUs, a.k.a. Train Service Providers / TSPs) and must decide
which services to schedule and how to adjust their timetables, subject to
capacity and safety constraints, while maximizing revenue and (optionally)
fairness across RUs.

## Decision variables

| Variable | Type | Meaning |
|---|---|---|
| Departure times | Real | The departure time at each intermediate stop of each service, within the IM's modification margin. |
| Scheduled mask | Discrete (binary) | Whether each service is scheduled (`1`) or dropped (`0`). |

The real variables are optimized by the metaheuristic (GSA or mealpy). The
discrete mask is derived from the real solution via the conflict-avoiding
heuristic (see [architecture.md](architecture.md#the-heuristic-schedule)).

## Constraints

- **Safety headway**: Two services sharing infrastructure segments cannot
  depart within `safe_headway` minutes of each other. Conflicts are detected
  by `robin`'s `ServiceScheduler.find_conflicts()` and encoded in the
  `ConflictMatrix`.
- **Boundary feasibility**: Each departure time must lie within
  `[reference - im_mod_margin, reference + im_mod_margin]` (default margin:
  60 minutes), propagated through travel and stop times.
- **Travel-time feasibility**: The updated travel time between consecutive
  stops cannot be shorter than the original (no speed-up). Likewise, stop
  times cannot be shortened.

## Revenue model

The revenue of a service is its **canon** (base revenue) minus penalties for
deviating from the requested timetable:

```
revenue(service) = canon - dt_penalty - sum(tt_penalties)
```

### Canon

The canon is synthesized by `RevenueSimulator` from the service's
characteristics:

```
canon = (distance_factor + capacity_factor + stations_factor) / 100
```

where:
- `distance_factor = 7 * total_line_distance_km`
- `capacity_factor = (alpha * rolling_stock_capacity) / 100 * 1.67`
- `stations_factor = 18 + (n_stations - 2) * 65 + 165`

The maximum penalty is `30%` of the canon, split into a departure-time
penalty (`35%` of the max) and travel-time penalties (the rest, distributed
across intermediate stops).

### Penalty function

Deviations are penalized by a damped-oscillation function bounded in
`[0, 1]`:

```
penalty(x, k) = 1 - e^(-k * x²) * (0.5 * cos(π * x) + 0.5)
```

where `x` is the normalized deviation (deviation / `im_mod_margin`) and `k`
is a per-RU sensitivity factor drawn from a log-uniform distribution.

- `x = 0` → no deviation → `penalty = 0`.
- `x > 0` → penalty grows, modulated by `k`.

### Importance

Each service is assigned a normalized importance weight within its RU group
(summing to 1 per RU). Importance is used by the fairness metrics.

## Fairness model

When fairness is considered, the objective combines revenue and a fairness
index:

```
objective = alpha * revenue / scale + beta * fairness * 100
```

where `alpha` and `beta` are configurable weights. CRAFT provides three
fairness indices via `FairnessMetrics`:

### Jain's index

```
J = (Σ r_u)² / (n * Σ r_u²)
```

where `r_u` is the ratio of scheduled importance to capacity for RU `u`.
Ranges from `1/n` (maximally unfair) to `1` (perfectly fair).

### Gini coefficient

```
G = (2 * Σ i * r_i) / (n * Σ r_i) - (n + 1) / n
```

where `r_i` are the sorted ratios. `G = 0` is perfect equality; the fairness
index is `1 - G`.

### Atkinson index

```
A = 1 - (geometric_mean / arithmetic_mean)
```

with an inequality-aversion parameter `epsilon`. The fairness index is
`1 - A`.

### Capacity-aware variants

The `jains_fairness_index`, `gini_fairness_index`, and
`atkinson_fairness_index` methods account for the RU capacities (share of
total services) and the importance weights of scheduled services, rather
than raw revenue values.
