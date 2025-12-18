# CADIS SROA Rep2 (Dung vs Dung+Reputation) — Prototype v7 (Q1-ready)

**Autor:** Omar Portilla Jaimes  
**Correo:** jorge.portilla2@unipamplona.edu.co  

Este proyecto Python empaquetado (`pip install -e .`) implementa la comparación entre:

- **Dung (grounded)**: consenso escéptico por argumentación abstracta.
- **Dung+Reputation (Hybrid)**: conserva grounded, pero convierte ataques en **derrotas efectivas** según un peso reputación–confianza.

Está diseñado como banco **sintético adversarial** (ruido + coalición + adversario) para estudiar robustez y gobernanza del filtrado.

---

## Filosofía de persistencia (Q1 + rendimiento)

Para evitar `/out` gigantes y I/O excesivo, por defecto el prototipo **NO** guarda trazas completas por evento en SQLite.  
En su lugar, por corrida guarda:

- `config.json` (incluye parámetros, semillas, autor, forgetting_rate y eta_equivalent),
- `code_fingerprint` + `config_fingerprint`,
- `checksums.json` (SHA256 de artefactos),
- **Excel paper-ready** (`run_summary.xlsx`, `metrics.xlsx`, etc.),
- Auditoría ligera (subconjunto de casos) si se activa.

La trazabilidad por evento queda disponible **solo si** se usa `--trace-level full --use-sqlite` (y no es el modo por defecto).

---

## Instalación

Desde la carpeta del proyecto (donde está `pyproject.toml`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

Verifica:

```powershell
cadis-sroa2 -h
cadis-sroa2 pipeline -h
```

---

## Comandos principales

### 1) Corrida única (compare) — genera `run_summary.xlsx`
```powershell
cadis-sroa2 compare --outdir out\compare_run --seed 7 --warmup 200 --test 200 --lam 1.25 --gamma 2.0 --omega 0.8 --forgetting 0.01 --trace-level audit --audit-max-cases 30 --audit-policy diff
```

### 2) Pipeline Q1 (baseline + sensibilidad + análisis)
```powershell
cadis-sroa2 pipeline --outdir out --n-samples 180 --seeds 7 11 13 17 19 --regimes R0_default R1_expert_degraded R3_more_adversarial --warmup 200 --test 200 --lam-min 1.0 --lam-max 1.8 --gamma-min 0.0 --gamma-max 4.0 --omega-min 0.0 --omega-max 1.0 --design-seed 123 --forgetting 0.01 --trace-level audit --audit-max-cases 30 --audit-policy diff
```

### 3) “Grid” de η por repetición (sin flag especial)
En el CLI se usa `--forgetting` como **tasa**. Si en el paper usas η como factor multiplicativo:
**η ≈ 1 − forgetting**.

Ejemplo (tres corridas independientes, cada una con su outdir):

```powershell
cadis-sroa2 pipeline --outdir out_eta_f000 --n-samples 180 --seeds 7 11 13 17 19 --regimes R0_default R1_expert_degraded R3_more_adversarial --warmup 200 --test 200 --lam-min 1.0 --lam-max 1.8 --gamma-min 0.0 --gamma-max 4.0 --omega-min 0.0 --omega-max 1.0 --design-seed 123 --forgetting 0.00 --trace-level audit --audit-max-cases 30 --audit-policy diff
cadis-sroa2 pipeline --outdir out_eta_f001 --n-samples 180 --seeds 7 11 13 17 19 --regimes R0_default R1_expert_degraded R3_more_adversarial --warmup 200 --test 200 --lam-min 1.0 --lam-max 1.8 --gamma-min 0.0 --gamma-max 4.0 --omega-min 0.0 --omega-max 1.0 --design-seed 123 --forgetting 0.01 --trace-level audit --audit-max-cases 30 --audit-policy diff
cadis-sroa2 pipeline --outdir out_eta_f005 --n-samples 180 --seeds 7 11 13 17 19 --regimes R0_default R1_expert_degraded R3_more_adversarial --warmup 200 --test 200 --lam-min 1.0 --lam-max 1.8 --gamma-min 0.0 --gamma-max 4.0 --omega-min 0.0 --omega-max 1.0 --design-seed 123 --forgetting 0.05 --trace-level audit --audit-max-cases 30 --audit-policy diff
```

---

## Estructura de salidas (paper-ready)

### compare (una corrida)
`<outdir>/run_summary.xlsx` con hojas:
- `config`
- `method_metrics`
- `case_aggregate`
- `audit_cases` (si aplica)

y además:
- `config.json`, `checksums.json`, `run_metrics.csv` (ligero)

### sensitivity (muchas corridas)
`<outdir>/sensitivity/`:
- `metrics.xlsx` (+ `metrics.csv`)
- `design.xlsx` (+ `design.csv`)
- `analysis/<regime>/prcc.xlsx`, `analysis/<regime>/pareto_front.xlsx`

### paper_outputs
`<outdir>/paper_outputs/`:
- `paper_metrics.xlsx`
- `paper_sensitivity.xlsx`
- `paper_prcc.xlsx`
- `paper_pareto.xlsx`

---

## Licencia
MIT


## Herramientas de análisis final (Q1)

En la carpeta `tools/` se incluye el script:

- `build_analisis_final_v3.py`

Este script lee directamente los resultados producidos por los *pipelines*:

- `out_eta_f001/`
- `out_eta_f005/`
- (o cualquier otro `outdir` que se le pase por línea de comandos)

y construye un archivo Excel integrado (por defecto `analisis_final_v3.xlsx`) con todas las tablas
"paper-ready" para el artículo en **Artificial Intelligence in Medicine**, incluyendo:

- métricas baseline por régimen,
- pruebas de McNemar exactas con *odds ratio* pareada y CI,
- deltas híbrido–puro con intervalos bootstrap,
- análisis de sensibilidad con binning de ruido,
- PRCC con *bootstrap* + permutaciones (ajustado por FDR),
- resumen de frentes de Pareto,
- índice de reproducibilidad (config + checksums + fingerprints),
- hoja de advertencias `MISSING_WARNINGS` totalmente transparente.

Uso típico (desde la raíz del proyecto):

```bash
python tools/build_analisis_final_v3.py \
    --project-root . \
    --outdirs out_eta_f001 out_eta_f005 \
    --output-xlsx analisis_final_v3.xlsx \
    --n-bootstrap 4000 \
    --prcc-bootstrap 1500 \
    --prcc-perm 1000 \
    --seed 123
```

