# HDB Resale Price Prediction

Predicting Singapore HDB resale flat prices using an ensemble of three Optuna-tuned gradient-boosted tree models with leakage-free spatial and categorical encoding.

---

## Results

| Validation Method | LightGBM | XGBoost | CatBoost | **Ensemble** |
| ----------------- | -------- | ------- | -------- | ------------ |
| 5-fold OOF        | 21,680   | 21,674  | 21,670   | **21,456**   |

- **MAPE ≈ 3.5%** — predictions are on average within SGD ~21,000 of the actual resale price
- **R² ≈ 0.977** — the model explains 97.7% of the variance in resale prices
- The 5-fold OOF result is the reliable estimate; the training set is used in full with no rows left out of evaluation

---

## Model Selection

### Why Gradient Boosted Decision Trees?

Three models were chosen — **LightGBM**, **XGBoost**, and **CatBoost** — all from the Gradient Boosted Decision Trees (GBDT) family.

GBDTs are the industry-standard choice for structured/tabular prediction problems:

| Property | Benefit for this problem |
| -------- | ------------------------ |
| Handles mixed feature types | Numeric distances, counts, binary flags, and encoded categoricals |
| No feature scaling required | Distances range 0–5 km; prices 200k–1.2 M — trees are scale-invariant |
| Robust to outliers | Premium flats and unusual transactions don't break splits the way they break linear models |
| Captures non-linear interactions | Floor × lease remaining, storey ratio × area — trees find these automatically |
| Native NaN handling | Several distance and school features have missing values |

Neural networks were considered but ruled out: on tabular datasets of this size (~150k rows, ~100 features), GBDTs consistently outperform them in both accuracy and training speed.

### Why These Three Specifically?

All three are GBDT variants, but each uses different regularisation and tree-building strategies, so their **prediction errors are partially uncorrelated** — the foundation for a useful ensemble.

| Model | Key differentiator |
| ----- | ------------------ |
| **LightGBM** | Leaf-wise tree growth; extremely fast; best with many leaves |
| **XGBoost** | Level-wise tree growth; strong L1/L2 regularisation; `gamma` controls split threshold |
| **CatBoost** | Ordered boosting; native handling of categorical features; `bagging_temperature` controls randomisation |

---

## Ensemble Methodology

### Why Ensemble?

A single model's prediction errors on individual rows are partly random. When two models disagree on a row, averaging them tends to be closer to the truth than either alone — provided the errors are not perfectly correlated.

The three models share the same features but differ in tree growth algorithm, regularisation mechanism, and subsampling strategy, giving genuine diversity confirmed by the ensemble consistently beating all individual models.

### How the Ensemble Works

Rather than a fixed average, weights are optimised using **Nelder-Mead** on OOF predictions:

```
minimise RMSE( y_true,  w₀·pred_lgb + w₁·pred_xgb + w₂·pred_cat )
subject to  w₀ + w₁ + w₂ = 1
```

| Split | LightGBM | XGBoost | CatBoost |
| ----- | -------- | ------- | -------- |
| 5-fold OOF | 0.32 | 0.23 | 0.45 |

CatBoost consistently receives the highest weight (~0.45) — Optuna tuning gave it a slight edge. LightGBM and XGBoost contribute meaningful diversity (combined 55%).

### Why Not Add Random Forest or Ridge?

Both were tested:
- **Ridge** (RMSE ≈ 55,000): far weaker than GBMs; assigned weight −0.01 by the optimiser — effectively excluded
- **Random Forest**: same tree-based paradigm; high error correlation with GBDTs, minimal diversity added

---

## Validation Strategy

### 5-Fold Out-of-Fold (OOF)

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

Every training row is predicted exactly **once** by a model that never trained on it — no leakage in the weight optimisation. Test predictions are the mean of 5 models (one per fold), which reduces prediction variance compared to a single model.

Target encoding and spatial encoding are both **recomputed inside each fold** using only that fold's training rows (see Encoding section).

---

## Hyperparameter Tuning

All three models were tuned using **Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler — a Bayesian optimisation method that builds a probabilistic model of the objective function and samples promising regions rather than searching randomly.

**Strategy:** single 80/20 inner split for each trial (fast proxy), then best params applied to full 5-fold OOF for honest final evaluation. 50 trials per model.

### LightGBM

| Parameter | Default | Tuned | Role |
| --------- | ------- | ----- | ---- |
| `learning_rate` | 0.05 | **0.010** | Smaller steps with 5000 estimators; better generalisation |
| `num_leaves` | 255 | **323** | More leaves → finer splits; leaf-wise growth uses this directly |
| `min_child_samples` | 20 | **43** | Minimum samples per leaf; prevents overfitting on sparse splits |
| `feature_fraction` | 0.80 | **0.567** | Column subsampling per tree; reduces correlation between trees |
| `bagging_fraction` | 0.80 | **0.889** | Row subsampling per iteration |
| `reg_alpha` | 0.10 | **0.00018** | L1 regularisation; nearly off — data is large enough |
| `reg_lambda` | 0.10 | **0.00039** | L2 regularisation; same conclusion |

### XGBoost

| Parameter | Default | Tuned | Role |
| --------- | ------- | ----- | ---- |
| `learning_rate` | 0.05 | **0.0105** | Slower learning; consistent with LGB finding |
| `max_depth` | 8 | **10** | Deeper trees capture more complex interactions |
| `min_child_weight` | 5 | **10** | Minimum sum of instance weight in a child; controls overfitting |
| `subsample` | 0.80 | **0.707** | Row subsampling fraction |
| `colsample_bytree` | 0.80 | **0.505** | Column subsampling fraction |
| `reg_alpha` | 0.10 | **0.0024** | L1 penalty |
| `reg_lambda` | 1.00 | **0.108** | L2 penalty; reduced — large dataset tolerates less shrinkage |
| `gamma` | 0 | **0.276** | Minimum loss reduction to make a split; acts as a pruning threshold |

### CatBoost

| Parameter | Default | Tuned | Role |
| --------- | ------- | ----- | ---- |
| `learning_rate` | 0.05 | **0.032** | Slower learning with 5000 iterations |
| `depth` | 8 | **10** | Deeper trees; CatBoost uses symmetric trees so depth is costly but expressive |
| `l2_leaf_reg` | 3 | **1.27** | L2 regularisation on leaf weights |
| `random_strength` | 1 | **1.54** | Randomness in split scoring; more exploration |
| `bagging_temperature` | 1 | **1.29** | Controls variance of bootstrap weights (Bayesian bootstrap) |
| `border_count` | 128 | **205** | Number of candidate split points per feature |

A warm-start follow-up extended `depth` to 12 and ran 20 more trials from the optimum — confirmed convergence with no improvement.

Both LGB and XGB settled on `lr ≈ 0.01` with `n_estimators=5000` + early stopping. This is a common Optuna finding: lower learning rates generalise better when compute allows.

---

## Feature Engineering

150,634 transactions were transformed into a richer feature set grouped below by type.

### Time Features

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `lease_remaining_years` | `99 − (Tranc_Year − lease_commence_date)` | Direct measure of remaining value in the 99-year HDB lease |
| `lease_remaining_pct` | `lease_remaining_years / 99` | Normalised version; easier for the model to compare across flat ages |
| `tranc_period` | `year × 12 + month` | Monotone time index capturing overall market trend |

Month (`Tranc_Month`) and cyclical features (`month_sin`, `month_cos`) were dropped in v14 after feature importance analysis showed no price signal.

### Storey / Floor Features

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `storey_ratio` | `mid_storey / max_floor_lvl` | Relative height matters more than absolute floor — floor 10 in a 12-storey block is high; in a 40-storey block it is not |
| `is_high_floor` | `mid_storey ≥ 20` | Binary premium flag; high floors command a discrete price jump in Singapore |
| `floor_band` | Binned storey (7 bands) | Ordinal compression of storey into price-relevant bands |

### Distance / Accessibility Features

Raw distances are **log-transformed** (`log1p`) to compress right-skewed distributions — the difference between 100 m and 200 m matters far more than the difference between 2,000 m and 2,100 m.

| Feature | Rationale |
| ------- | --------- |
| `log_mrt_dist` | MRT proximity is the single strongest price driver in Singapore |
| `log_mall_dist` | Retail accessibility |
| `log_hawker_dist` | Hawker centres are a cultural amenity unique to Singapore |
| `accessibility_score` | Weighted composite: `0.4×MRT + 0.2×mall + 0.2×hawker` — single summary of overall connectivity |

`log_bus_dist`, `log_pri_dist`, and `log_sec_dist` were dropped in v14 (low feature importance).

### Interaction Features

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `area_x_storey` | `floor_area_sqm × mid_storey` | Large high-floor flats command a multiplicative premium |
| `area_x_lease_rem` | `floor_area_sqm × lease_remaining_years` | Bigger flats with longer leases are disproportionately valuable |
| `storey_x_lease_rem` | `mid_storey × lease_remaining_years` | High-floor flats depreciate more sharply as lease shortens |
| `year_completed_x_floor_area` | `year_completed × floor_area_sqm` | Newer large flats occupy a premium segment |

### School Quality Features

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `school_quality` | `cutoff_point + affiliation × 10` | Secondary school selectivity is a known HDB price signal in popular districts |
| `pri_school_quality` | `affiliation × 10 + 1/(distance + 1)` | Combines school prestige with proximity |

### Dropped / Excluded Features

PCA analysis identified redundant pairs; v14 feature importance removed additional low-signal features:

| Dropped | Reason |
| ------- | ------ |
| `floor_area_sqft` | Exact unit conversion of `floor_area_sqm` (r = 1.0) |
| `hdb_age` | Near-perfect negative of `year_completed` |
| `lower`, `upper`, `mid` | `mid_storey = (lower + upper) / 2`; `mid` is identical |
| Rental unit counts (`1room_rental`, …) | Mean gain < 0.10% across all models |
| Unit mix sold counts (`1room_sold`, …) | Redundant with `flat_type` |
| Binary building flags (`residential`, `commercial`, …) | No price signal |
| `month_sin`, `month_cos`, `Tranc_Month` | No seasonal price signal detected |
| `Latitude`, `Longitude` | Raw coordinates absorbed by spatial encoding (still used internally) |

---

## Encoding

### Target Encoding — `town`, `planning_area`

These nominal location columns carry a direct price signal. Label encoding would assign arbitrary ordinal ranks. **Smoothed mean target encoding** converts each category to the mean price of transactions in that area, shrunk toward the global mean for small groups:

```
encoded = global_mean × (1 − s) + group_mean × s
where  s = sigmoid(count − 10)   ← shrinks small groups toward global mean
```

#### Why It Is Leak-Free

The critical implementation detail is **when** the encoding statistics are computed. In the final pipeline (v11+), encoding is done **inside the fold loop**:

```python
# Inside each fold:
df_tr_fold  = train.iloc[tr_idx]   # ~80% of training data
df_val_fold = train.iloc[val_idx]  # ~20%, never seen by encoder

# Statistics computed from training fold only
global_mean = df_tr_fold['resale_price'].mean()
enc = df_tr_fold.groupby('town')['resale_price'].agg(['mean', 'count'])
# ... apply smoothing ...

# Validation rows are mapped to training statistics — never contributed their own price
df_val_fold['town_te'] = df_val_fold['town'].map(enc).fillna(global_mean)
```

Each fold's validation rows are predicted using encoding statistics that were computed without them. The test set is also mapped to training-fold statistics only. No row's price ever leaks into its own encoding.

The earlier `final.py` version computed encoding on the full training set before splitting, which was a subtle leak: validation rows' own prices contributed to their town's encoding. v11 closed this gap.

### Label Encoding — `flat_type`, `flat_model`, `mrt_name`, `pri_sch_name`, `sec_sch_name`

Label encoding assigns integer indices to each category. Because the target is not involved, this is computed globally (fitting on train + test combined to handle unseen categories) and is safe at all times. Used here because all three tree models handle ordinal integers natively and it is simpler than one-hot for high-cardinality columns.

---

## Spatial Encoding (KD-Tree)

### Why Geohash Was Dropped

**v12** introduced geohash spatial encoding: each transaction is assigned to a geohash cell (~150 m × 150 m at precision 7) and the smoothed mean price of all transactions in that cell is used as a feature.

The problem is the **cell boundary discontinuity**. Geohash divides space into discrete rectangles. Two transactions 10 m apart on opposite sides of a cell boundary receive completely different encodings even though their location-based prices should be nearly identical. Expanding to 9 adjacent cells (`geo6_nbr_price`) softened but did not remove this discontinuity — the 3×3 patch still has hard boundaries.

### Why KD-Tree (v13+)

**cKDTree** (`scipy.spatial`) replaces geohash with continuous radius-based aggregation:

1. Build a KD-tree from training-fold coordinates (Latitude, Longitude converted to metres).
2. For each query point, find **all training transactions within r metres**.
3. Compute the smoothed mean price of those neighbours.

This gives a **fully continuous price surface** — no grid cells, no boundaries. A transaction at any location smoothly interpolates the prices of nearby training transactions.

```python
# Built from training-fold coordinates only → leakage-free
tree = cKDTree(coords_train)

# For each query point, radius search in O(n log n)
neighbours = tree.query_ball_point(query_coords, r=500, workers=-1)
smoothed_mean = sigmoid_smooth(len(neighbours), mean(prices[neighbours]), global_mean)
```

No external library required — `scipy` is already a dependency.

Three spatial features are added per fold:

| Feature | Radius | What it captures |
| ------- | ------ | ---------------- |
| `spatial_500m_te` | 500 m | Block-level price environment (~immediate neighbourhood) |
| `spatial_2000m_te` | 2000 m | District-level price environment |
| `spatial_500m_psm` | 500 m | Mean price-per-sqm within 500 m (size-normalised) |

All three are computed from training-fold data only — the same leakage-free discipline as target encoding.

---

## Pipeline Versions

| Version | Key change | Ensemble OOF RMSE | Kaggle RMSE |
| ------- | ---------- | ----------------- | -------- |
| v1 | Baseline (80/20 split, LGB/XGB/CAT) | 21,220 | 22,067 |
| v2 | 5-fold CV | 21,211 | 21,498
| v3 | PCA-informed: dropped 5 redundant cols + `year_completed_x_floor_area` | 21,195 |
| v4 | Target encoding for `town`/`planning_area` | 21,177 |
| v5 | Added CBD distance, price trend, building age *(reverted — redundant)* | 21,195 |
| v6 | 5-fold OOF stacking | 21,542 | 21,466 | 
| v7 | CatBoost Optuna HPO (50 trials) | 21,525 |
| v8 | CatBoost warm-start, depth ≤ 12 — confirmed convergence | 21,525 |
| v9 | LightGBM + XGBoost Optuna HPO (50 trials each) | 21,459 |
| v10 | Dual output: 80/20 + 5-fold OOF | 80/20: 21,105<br>5-fold: 21,456 | 80/20: 21,579<br> 5-Fold: 21,413
| v11 | Leak-free target encoding (computed inside fold loop) | 21,461 |
| v12 | Geohash spatial encoding (4 features) | 21,680 | 21,714
| v13 | KD-tree replaces geohash (continuous, no boundary artefacts) | 21,456 | 21,396
| **v14** | Dropped 31 low-importance features; 5-fold OOF only | 21,369 | 21,284

---

## Pros and Cons

### Pros

| Strength | Detail |
| -------- | ------ |
| **High accuracy** | R² = 0.977, MAPE = 3.5% — predictions typically within SGD 21,000 of true price |
| **Robust validation** | 5-fold OOF ensures every row is evaluated out-of-sample; no data is wasted |
| **All models tuned** | Optuna TPE with 50 trials per model; warm-start confirms convergence |
| **Leak-free encoding** | Target and spatial encoding stats computed from training-fold data only; test stats imputed from training distribution |
| **Continuous spatial features** | KD-tree radius search provides a smooth price surface with no grid-boundary discontinuities |
| **Diverse ensemble** | Three different GBDT implementations with partially uncorrelated errors; Nelder-Mead finds optimal weights |
| **Interpretable features** | All engineered features have clear domain meaning; no black-box transformations |

### Cons

| Limitation | Detail |
| ---------- | ------ |
| **No macro market signal** | HDB Resale Price Index (RPI) not incorporated; `tranc_period` is a coarse proxy for market cycle timing |
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
│   ├── 00-PCA-analysis.ipynb           ← feature group analysis, redundancy detection
│   └── 00-prelimanary-analysis.ipynb
├── script/
│   ├── hdb_ml_pipeline_v14.py          ← current version (use this)
│   ├── hdb_ml_pipeline_v13.py          ← KD-tree spatial encoding
│   ├── hdb_ml_pipeline_v12.py          ← geohash spatial (superseded)
│   ├── hdb_ml_pipeline_v11.py          ← leak-free target encoding
│   └── hdb_ml_pipeline_v1..v10.py      ← earlier iterations
└── submission/
    └── submission_v14_5fold.csv         ← latest (recommended)
```

## Running the Pipeline

```bash
cd script
python hdb_ml_pipeline_v14.py
```

**Requirements:** `pandas numpy scikit-learn lightgbm xgboost catboost scipy optuna`

**Runtime:** approximately 35–45 minutes on Apple M-series CPU (KD-tree radius search adds ~5 minutes per fold).

The script outputs `submission_v14_5fold.csv` and prints a feature importance table averaged across all 5 folds.
