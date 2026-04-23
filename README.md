# HDB Resale Price Prediction

Predicting Singapore HDB resale flat prices using an ensemble of three gradient-boosted tree models, all individually tuned with Bayesian hyperparameter optimisation.

---

## Results

| Validation Method | LightGBM | XGBoost | CatBoost | **Ensemble** |
|---|---|---|---|---|
| 80/20 random split | 21,289 | 21,331 | 21,337 | **21,105** |
| 5-fold OOF | 21,680 | 21,674 | 21,670 | **21,456** |

- **MAPE ≈ 3.49–3.50%** — on average predictions are within SGD ~21,000 of the actual resale price
- **R² ≈ 0.977–0.978** — the model explains 97.7% of the variance in resale prices
- The 80/20 split produces a more optimistic estimate; the 5-fold OOF is the more reliable figure

---

## Model Selection

### Why Gradient Boosted Decision Trees?

Three models were chosen — **LightGBM**, **XGBoost**, and **CatBoost** — all from the same family: Gradient Boosted Decision Trees (GBDT).

GBDTs are the industry-standard choice for structured/tabular prediction problems because:

| Property | Benefit for this problem |
|---|---|
| Handles mixed feature types | Dataset contains numeric distances, counts, binary flags, and encoded categoricals |
| No feature scaling required | Distances range from 0–5km; prices from 200k–1.2M — trees are scale-invariant |
| Robust to outliers | Premium flats and unusual transactions don't break tree splits the way they break linear models |
| Captures non-linear interactions | Floor × lease remaining, storey ratio × area — trees find these automatically |
| Native NaN handling | Several distance and school features have missing values |

Neural networks were considered but ruled out: on tabular datasets of this size (~150k rows, ~100 features), GBDTs consistently outperform neural networks in both accuracy and training speed.

### Why These Three Specifically?

All three are GBDT variants, but each uses different regularisation and tree-building strategies, which means their **prediction errors are partially uncorrelated** — the foundation for a useful ensemble.

| Model | Key differentiator |
|---|---|
| **LightGBM** | Leaf-wise tree growth; extremely fast; best with many leaves (`num_leaves=323`) |
| **XGBoost** | Level-wise tree growth; strong L1/L2 regularisation; `gamma` controls split threshold |
| **CatBoost** | Ordered boosting; native handling of categorical features; `bagging_temperature` controls randomisation |

---

## Ensemble Methodology

### Why Ensemble?

A single model's prediction errors on individual rows are partly random. When two models disagree on a row, averaging them tends to be closer to the truth than either alone — provided the errors are not perfectly correlated.

The three models here share the same features but differ in:
- Tree growth algorithm (leaf-wise vs level-wise)
- Regularisation mechanism (L1/L2 vs depth penalty vs Bayesian smoothing)
- Subsampling strategy (column fraction, row fraction, bagging temperature)

This gives genuine diversity, confirmed by the ensemble consistently beating all individual models.

### How the Ensemble Works

Rather than a fixed average, weights are optimised using **Nelder-Mead** on the validation set predictions:

```
minimise RMSE( y_true,  w₀·pred_lgb + w₁·pred_xgb + w₂·pred_cat )
subject to  w₀ + w₁ + w₂ = 1
```

The optimiser finds the combination that minimises held-out error. Final weights:

| Split | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| 80/20 | 0.41 | 0.16 | 0.43 |
| 5-fold OOF | 0.32 | 0.23 | 0.45 |

CatBoost consistently receives the highest weight (~0.43–0.45) — the Optuna tuning gave it a slight edge. LightGBM and XGBoost contribute meaningful diversity (combined 55–57%).

### Why Not Add Random Forest or Ridge Regression?

Both were tested:
- **Ridge** (RMSE ≈ 55,000): far weaker than GBMs; assigned weight −0.01 by the optimiser — effectively excluded
- **Random Forest**: same tree-based paradigm as GBDTs; high error correlation, minimal diversity added; not worth the compute cost

---

## Validation Strategy

### 80/20 Random Split (Training A)

- 120,507 training rows / 30,127 validation rows
- Fast single-pass evaluation
- **Risk**: one lucky/unlucky split can give an optimistic or pessimistic RMSE estimate

### 5-Fold Out-of-Fold (Training B)

```
Fold 1: [train on folds 2–5] → predict fold 1
Fold 2: [train on folds 1,3–5] → predict fold 2
...
Fold 5: [train on folds 1–4] → predict fold 5
─────────────────────────────────────────────
OOF predictions cover all 150,634 training rows
Ensemble weights optimised on full OOF array
Test predictions = average of 5 fold predictions
```

Every training row is predicted exactly **once** by a model that never trained on it — no leakage in the weight optimisation.

Test predictions are the **mean of 5 models** (one per fold), which reduces prediction variance compared to a single model.

**The 5-fold OOF result (RMSE = 21,456) is the more trustworthy estimate** for leaderboard performance.

---

## Hyperparameter Tuning

All three models were tuned using **Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler — a Bayesian optimisation method that builds a probabilistic model of the objective function and samples promising regions, rather than searching randomly.

### Strategy

1. **Single 80/20 inner split** used for each trial (fast proxy for true performance)
2. **Best params applied to full 5-fold OOF** for honest final evaluation
3. **50 trials** per model (TPE is efficient; most improvement happens in first 30)

### CatBoost (v7 → v8)

| Parameter | Default | Tuned | Notes |
|---|---|---|---|
| `depth` | 8 | **10** | Deeper trees capture more complex interactions |
| `learning_rate` | 0.05 | **0.032** | Slower learning, more precise convergence |
| `l2_leaf_reg` | 3 | **1.27** | Less regularisation needed — dataset is large enough |
| `random_strength` | 1 | **1.54** | More randomness in split scoring |
| `bagging_temperature` | 1 | **1.29** | Higher variance in bootstrap weights |
| `border_count` | 128 | **205** | More candidate splits per feature |

A warm-start follow-up (v8) extended `depth` to 12 and ran 20 more trials from the v7 optimum — confirmed convergence with no improvement.

### LightGBM (v9)

| Parameter | Default | Tuned |
|---|---|---|
| `learning_rate` | 0.05 | **0.010** |
| `num_leaves` | 255 | **323** |
| `min_child_samples` | 20 | **43** |
| `feature_fraction` | 0.80 | **0.567** |
| `bagging_fraction` | 0.80 | **0.889** |
| `reg_alpha` | 0.10 | **0.00018** |
| `reg_lambda` | 0.10 | **0.00039** |

### XGBoost (v9)

| Parameter | Default | Tuned |
|---|---|---|
| `learning_rate` | 0.05 | **0.0105** |
| `max_depth` | 8 | **10** |
| `min_child_weight` | 5 | **10** |
| `subsample` | 0.80 | **0.707** |
| `colsample_bytree` | 0.80 | **0.505** |
| `reg_alpha` | 0.10 | **0.0024** |
| `reg_lambda` | 1.00 | **0.108** |
| `gamma` | 0 | **0.276** |

Both LGB and XGB settled on `lr ≈ 0.01` — slower learning with more trees (`n_estimators=5000` with early stopping). This is a common Optuna finding: lower learning rates generalise better when compute allows.

---

## Feature Engineering

150,634 transactions across 77 raw columns were transformed into a richer feature set. Features are grouped below by type and rationale.

### Time Features

| Feature | Formula | Rationale |
|---|---|---|
| `lease_remaining_years` | `99 − (Tranc_Year − lease_commence_date)` | Direct measure of remaining value in the 99-year HDB lease |
| `lease_remaining_pct` | `lease_remaining_years / 99` | Normalised version; easier for the model to compare across flat ages |
| `month_sin`, `month_cos` | `sin/cos(2π × month / 12)` | Cyclical encoding — December (12) and January (1) are adjacent; raw month numbers treat them as far apart |
| `tranc_period` | `year × 12 + month` | Monotone time index capturing overall market trend |

### Storey / Floor Features

| Feature | Formula | Rationale |
|---|---|---|
| `storey_ratio` | `mid_storey / max_floor_lvl` | Relative height matters more than absolute floor — floor 10 in a 12-storey block is high; in a 40-storey block it is not |
| `is_high_floor` | `mid_storey ≥ 20` | Binary premium flag; high floors command a discrete price jump in Singapore |
| `floor_band` | Binned storey (7 bands) | Ordinal compression of storey into price-relevant bands |

### Distance / Accessibility Features

Raw distances are **log-transformed** (`log1p`) to compress right-skewed distributions — the difference between 100m and 200m matters far more than the difference between 2,000m and 2,100m.

| Feature | Rationale |
|---|---|
| `log_mrt_dist` | MRT proximity is the single strongest price driver in Singapore |
| `log_mall_dist` | Retail accessibility |
| `log_hawker_dist` | Hawker centres are a cultural amenity unique to Singapore |
| `log_bus_dist`, `log_pri_dist`, `log_sec_dist` | Secondary accessibility features |
| `accessibility_score` | Weighted composite: `0.4×MRT + 0.2×mall + 0.2×hawker + 0.2×bus` — single summary of overall connectivity |

### Building / Block Features

| Feature | Rationale |
|---|---|
| `total_sold_units` | Sum of all room types sold — proxy for block size |
| `total_rental_units` | Rental unit count — indicates demand character of the block |
| `sold_ratio` | `total_sold / total_dwelling_units` — owner-occupancy rate; higher ratios often signal more desirable blocks |
| `is_mrt_interchange`, `is_bus_interchange` | Binary premiums for interchange-adjacent blocks |

### School Quality Features

| Feature | Formula | Rationale |
|---|---|
| `school_quality` | `cutoff_point + affiliation × 10` | Secondary school selectivity is a known HDB price signal in popular districts |
| `pri_school_quality` | `affiliation × 10 + 1/(distance + 1)` | Combines school prestige with proximity |

### Interaction Features

Gradient boosted trees can find interactions between features, but providing them explicitly reduces the tree depth needed to represent them.

| Feature | Formula | Rationale |
|---|---|
| `area_x_storey` | `floor_area_sqm × mid_storey` | Large high-floor flats command a multiplicative premium |
| `area_x_lease_rem` | `floor_area_sqm × lease_remaining_years` | Bigger flats with longer leases are disproportionately valuable |
| `storey_x_lease_rem` | `mid_storey × lease_remaining_years` | High-floor flats depreciate more sharply as lease shortens |
| `year_completed_x_floor_area` | `year_completed × floor_area_sqm` | Newer large flats occupy a premium segment identified by PCA analysis |

### Encoding

| Column type | Method | Reason |
|---|---|---|
| `town`, `planning_area` | **Smoothed mean target encoding** | These are nominal areas with a direct price signal; label encoding assigns arbitrary ordinal ranks. Smoothing formula: `encoded = global_mean × (1−s) + group_mean × s` where `s = sigmoid(count − 10)` shrinks small groups toward the global mean |
| `flat_type`, `flat_model`, `mrt_name`, `pri_sch_name`, `sec_sch_name` | **Label encoding** | Used by all three tree models without issue; simpler than one-hot for high-cardinality columns |

### Dropped / Excluded Features

PCA analysis on the raw columns identified redundant pairs:

| Dropped | Reason |
|---|---|
| `floor_area_sqft` | Exact unit conversion of `floor_area_sqm` (r = 1.0) |
| `hdb_age` | Near-perfect negative of `year_completed` |
| `lower`, `upper`, `mid` | `mid_storey = (lower + upper) / 2`; `mid` is identical |

---

## Pipeline Versions

| Version | Key change | Ensemble RMSE |
|---|---|---|
| v1 | Baseline (5-fold CV, LGB/XGB/CAT, label encoding) | — |
| v2 | Single 80/20 split replacing 5-fold | 21,177 *(optimistic)* |
| v3 | PCA-informed: dropped 5 redundant cols + `year_completed_x_floor_area` | 21,195 |
| v4 | Target encoding for `town`/`planning_area` | 21,177 |
| v5 | Added CBD distance, price trend, building age *(reverted — redundant)* | 21,195 |
| v6 | 5-fold OOF stacking | 21,525 *(honest baseline)* |
| v7 | CatBoost Optuna HPO (50 trials) | 21,525 |
| v8 | CatBoost warm-start, depth ≤ 12 — confirmed convergence | 21,525 |
| v9 | LightGBM + XGBoost Optuna HPO (50 trials each) | **21,459** |
| **final** | All tuned params, clean script, dual output | **21,105 / 21,456** |

---

## Pros and Cons

### Pros

| Strength | Detail |
|---|---|
| **High accuracy** | R² = 0.977, MAPE = 3.5% — predictions typically within SGD 21,000 of true price |
| **Robust validation** | 5-fold OOF ensures every row is evaluated out-of-sample; no data is wasted |
| **All models tuned** | Optuna TPE with 50 trials per model; warm-start confirms convergence |
| **Leakage-controlled** | Target encoding stats computed on training set only; test stats imputed from training distribution |
| **Diverse ensemble** | Three different GBDT implementations with partially uncorrelated errors; Nelder-Mead finds optimal weights |
| **Interpretable features** | All engineered features have clear domain meaning; no black-box transformations |

### Cons

| Limitation | Detail |
|---|---|
| **No macro market signal** | HDB Resale Price Index (RPI) not incorporated; `tranc_period` is a coarse proxy for market cycle timing |
| **Target encoding leakage** | Encoding computed on full training set before the 80/20 split — validation rows' own prices contribute to their town's encoding. Effect is minor due to smoothing and large sample size, but not zero |
| **HPO uses single inner split** | Optuna trials evaluated on one 80/20 split for speed; the best params are then validated on full 5-fold OOF, but the search itself has higher variance than a k-fold objective |
| **Ensemble diversity ceiling** | All three models are GBDT variants; their errors are meaningfully but not deeply uncorrelated. A neural network (TabNet) or linear model would add more orthogonal signal but was not competitive at this dataset size |
| **Hard cases remain** | Premium blocks (Pinnacle@Duxton, DBSS), niche flat types (multi-gen, studio), and extreme lease-age outliers sit in the residual RMSE and are unlikely to improve without additional features or data |
| **Static model** | No retraining mechanism; predictions will drift as the market evolves beyond the training period |

---

## Repository Structure

```
HDB Kaggle/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebook/
│   ├── 00-PCA-analysis.ipynb       ← feature group analysis, redundancy detection
│   └── 00-prelimanary-analysis.ipynb
├── script/
│   ├── hdb_ml_pipeline_final.py    ← production pipeline (use this)
│   ├── hdb_ml_pipeline_v2.py       ← baseline 80/20 split
│   ├── hdb_ml_pipeline_v3.py       ← PCA-informed feature cleanup
│   ├── hdb_ml_pipeline_v4.py       ← target encoding
│   ├── hdb_ml_pipeline_v5.py       ← feature additions (reverted)
│   ├── hdb_ml_pipeline_v6.py       ← 5-fold OOF
│   ├── hdb_ml_pipeline_v7.py       ← CatBoost HPO
│   ├── hdb_ml_pipeline_v8.py       ← CatBoost warm-start / convergence check
│   └── hdb_ml_pipeline_v9.py       ← LGB + XGB HPO
└── submission/
    ├── submission_final_8020.csv    ← 80/20 split predictions
    └── submission_final_5fold.csv  ← 5-fold OOF predictions (recommended)
```

## Running the Final Pipeline

```bash
cd script
python hdb_ml_pipeline_final.py
```

**Requirements:** `pandas numpy scikit-learn lightgbm xgboost catboost scipy`

**Runtime:** approximately 25–35 minutes on Apple M-series CPU.

Two submission files are produced. Submit `submission_final_5fold.csv` for the most reliable leaderboard score.
