# HDB Resale Price Prediction

Predicting Singapore HDB resale flat prices using an ensemble of three gradient-boosted tree models with leakage-free spatial and categorical encoding. Hyperparameters tuned with Optuna (3-fold inner CV), evaluated on 5-fold out-of-fold predictions.

---

## Results

| Model | OOF RMSE | OOF MAE | OOF MAPE | OOF R² |
| ----- | -------- | ------- | -------- | ------ |
| LightGBM | 21,525 | 15,443 | 3.50% | 0.9774 |
| XGBoost | 21,543 | 15,441 | 3.50% | 0.9774 |
| CatBoost | 21,531 | 15,448 | 3.51% | 0.9774 |
| **Ensemble** (L=0.38, X=0.18, C=0.45) | **21,325** | **15,292** | **3.47%** | **0.9779** |

- Predictions are on average within **SGD ~15,300** of the actual resale price (MAE)
- The model explains **97.8%** of the variance in resale prices
- All numbers are 5-fold OOF — every row is evaluated by a model that never trained on it
- Kaggle leaderboard score: **21,225** (public test set, v20)

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

Neural networks were ruled out: on tabular datasets of this size (~150k rows, ~60 features), GBDTs consistently outperform them in both accuracy and training speed.

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

The three models share the same features but differ in tree growth algorithm, regularisation mechanism, and subsampling strategy, giving genuine diversity confirmed by the ensemble consistently beating all individual models by ~200 RMSE points.

### How the Ensemble Works

Rather than a fixed average, weights are optimised using **Nelder-Mead** on OOF predictions:

```
minimise RMSE( y_true,  w₀·pred_lgb + w₁·pred_xgb + w₂·pred_cat )
subject to  w₀ + w₁ + w₂ = 1
```

| Version | LightGBM | XGBoost | CatBoost |
| ------- | -------- | ------- | -------- |
| v20 (current) | 0.38 | 0.18 | 0.45 |

CatBoost receives the highest weight — its ordered boosting gives it a consistent edge. All three contribute meaningfully; Nelder-Mead converges to non-trivial weights for each.

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

**Strategy (v15+):** 3-fold inner CV for each Optuna trial (more stable than single 80/20 split), 50 trials per model. Best params hardcoded into production scripts. v20 uses params from the v19 Optuna run.

### LightGBM

| Parameter | Tuned | Role |
| --------- | ----- | ---- |
| `learning_rate` | **0.0141** | Slow learning with 5000 estimators; better generalisation |
| `num_leaves` | **230** | Controls tree complexity; leaf-wise growth uses this directly |
| `min_child_samples` | **118** | Minimum samples per leaf; prevents overfitting on sparse splits |
| `feature_fraction` | **0.404** | Column subsampling per tree; reduces correlation between trees |
| `bagging_fraction` | **0.978** | Row subsampling per iteration |
| `reg_alpha` | **0.056** | L1 regularisation |
| `reg_lambda` | **0.00043** | L2 regularisation |

### XGBoost

| Parameter | Tuned | Role |
| --------- | ----- | ---- |
| `learning_rate` | **0.0107** | Consistent with LGB finding — slow + deep |
| `max_depth` | **11** | Deeper trees capture more complex interactions |
| `min_child_weight` | **29** | Minimum sum of instance weight in a child; controls overfitting |
| `subsample` | **0.694** | Row subsampling fraction |
| `colsample_bytree` | **0.403** | Column subsampling fraction |
| `reg_alpha` | **0.0092** | L1 penalty |
| `reg_lambda` | **0.0113** | L2 penalty |
| `gamma` | **2.662** | Minimum loss reduction to make a split; strong pruning threshold |

### CatBoost

| Parameter | Tuned | Role |
| --------- | ----- | ---- |
| `learning_rate` | **0.0369** | Slower learning with 5000 iterations |
| `depth` | **9** | Symmetric tree depth; shallower than XGB but each level is full-width |
| `l2_leaf_reg` | **0.047** | L2 regularisation on leaf weights |
| `random_strength` | **0.730** | Randomness in split scoring |
| `bagging_temperature` | **0.581** | Controls variance of bootstrap weights (Bayesian bootstrap) |
| `border_count` | **192** | Number of candidate split points per feature |

Both LGB and XGB settled on `lr ≈ 0.01` with `n_estimators=5000` + early stopping — a common Optuna finding: lower learning rates generalise better when compute allows.

---

## Feature Engineering

150,634 transactions were transformed into a richer feature set grouped below by type. **59 features total per fold** (53 base + 2 target-encoded + 4 spatial).

### Time Features

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `lease_remaining_years` | `99 − (Tranc_Year − lease_commence_date)` | Direct measure of remaining value in the 99-year HDB lease |
| `lease_remaining_pct` | `lease_remaining_years / 99` | Normalised version; easier for the model to compare across flat ages |
| `tranc_period` | `year × 12 + month` | Monotone time index capturing overall market trend |

`Tranc_Month` dropped — no seasonal price signal detected.

### Storey / Floor Features

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `storey_ratio` | `mid_storey / max_floor_lvl` | Relative height matters more than absolute floor — floor 10 in a 12-storey block is high; in a 40-storey block it is not |
| `floor_band` | Binned storey (7 bands: 1–5, 6–10, 11–15, 16–20, 21–30, 31–50, 51+) | Ordinal compression of storey into price-relevant bands |

### Distance / Accessibility Features

Raw distances are **log-transformed** (`log1p`) to compress right-skewed distributions — the difference between 100 m and 200 m matters far more than the difference between 2,000 m and 2,100 m. All six raw distance columns are dropped in favour of their log versions.

| Feature | Replaces | Rationale |
| ------- | -------- | --------- |
| `log_mrt_dist` | `mrt_nearest_distance` | MRT proximity is the single strongest price driver in Singapore |
| `log_mall_dist` | `Mall_Nearest_Distance` | Retail accessibility |
| `log_hawker_dist` | `Hawker_Nearest_Distance` | Hawker centres are a cultural amenity unique to Singapore |
| `log_bus_dist` | `bus_stop_nearest_distance` | Transit granularity beyond MRT |
| `log_pri_sch_dist` | `pri_sch_nearest_distance` | Primary school proximity affects registration priority |
| `log_sec_sch_dist` | `sec_sch_nearest_dist` | Secondary school proximity |
| `accessibility_score` | — | Weighted composite: `0.4×MRT + 0.2×mall + 0.2×hawker` |

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
| `pri_school_quality` | `pri_sch_affiliation × 10 + 1/(pri_sch_nearest_distance + 1)` | Combines school prestige with proximity |

**`school_quality`** — `cutoff_point` is the minimum PSLE aggregate for entry into the nearest secondary school; higher means more selective. `affiliation` is a binary flag for branded secondary pipeline schools (e.g. Nanyang Primary → Nanyang Girls' High). The `× 10` multiplier brings the binary flag into the same magnitude as the cut-off score range (~4–25).

**`pri_school_quality`** — primary school distance matters more than secondary because Singapore's registration system grants priority admission to families within 1 km. The proximity term `1/(distance + 1)` decays from 1.0 at the doorstep to near-zero beyond a few hundred metres; `affiliation × 10` flags schools with a prestigious secondary pipeline.

### Feature Importance (v20, 5-fold average gain)

Top features by mean gain across all three models:

| Rank | Feature | Mean% | LGB% | XGB% | CAT% |
| ---- | ------- | ----- | ---- | ---- | ---- |
| 1 | `flat_type_enc` | 14.48 | 7.74 | 24.39 | 11.30 |
| 2 | `area_x_lease_rem` | 11.07 | 14.32 | 5.95 | 12.93 |
| 3 | `spatial_500m_psm` | 11.03 | 13.21 | 6.41 | 13.47 |
| 4 | `year_completed_x_floor_area` | 10.92 | 17.64 | 8.01 | 7.11 |
| 5 | `spatial_2000m_psm` | 6.97 | 6.10 | 6.66 | 8.13 |
| 6 | `floor_area_sqm` | 6.94 | 8.91 | 4.33 | 7.56 |
| 7 | `spatial_500m_te` | 3.10 | 4.15 | 2.62 | 2.54 |
| 8 | `Hawker_Within_2km` | 2.41 | 1.57 | 4.91 | 0.77 |
| 9 | `area_x_storey` | 2.24 | 2.47 | 1.31 | 2.95 |
| 10 | `tranc_period` | 2.03 | 1.80 | 0.40 | 3.87 |

Spatial PSM features (price-per-sqm within 500 m and 2000 m) rank 3rd and 5th — capturing neighbourhood size-normalised price is highly predictive.

### Dropped / Excluded Features

| Dropped | Reason |
| ------- | ------ |
| `floor_area_sqft` | Exact unit conversion of `floor_area_sqm` (r = 1.0) |
| `hdb_age` | Near-perfect negative of `year_completed` |
| `lower`, `upper`, `mid` | `mid_storey = (lower + upper) / 2`; `mid` is identical |
| 6 raw distance columns | Replaced by log-transformed versions |
| Rental unit counts (`1room_rental`, …) | Mean gain < 0.10% across all models |
| Unit mix sold counts (`1room_sold`, …) | Redundant with `flat_type` |
| Binary building flags (`residential`, `commercial`, …) | No price signal |
| `Tranc_Month` | No seasonal price signal detected |
| `Latitude`, `Longitude` | Raw coordinates absorbed by spatial encoding (used internally for KD-tree) |

---

## Encoding

### Target Encoding — `town`, `planning_area`

These nominal location columns carry a direct price signal. Label encoding would assign arbitrary ordinal ranks. **Smoothed mean target encoding** converts each category to the mean price of transactions in that area, shrunk toward the global mean for small groups:

```
encoded = global_mean × (1 − s) + group_mean × s
where  s = sigmoid(count − 10)   ← shrinks small groups toward global mean
```

#### Why It Is Leak-Free (Val/Test)

The critical implementation detail is **when** the encoding statistics are computed. Encoding is done **inside the fold loop**:

```python
# Inside each fold:
df_tr_fold  = train.iloc[tr_idx]   # ~80% of training data
df_val_fold = train.iloc[val_idx]  # ~20%, never seen by encoder

# Statistics computed from training fold only
global_mean = df_tr_fold['resale_price'].mean()
enc = df_tr_fold.groupby('town')['resale_price'].agg(['mean', 'count'])
# ... apply smoothing ...

# Validation rows mapped to training statistics — never contributed their own price
df_val_fold['town_te'] = df_val_fold['town'].map(enc).fillna(global_mean)
```

Each fold's validation rows are predicted using encoding statistics computed without them. The test set is also mapped to training-fold statistics only.

#### Train-Fold Self-Leakage

A subtler form of leakage exists on the **training side**: when applying the encoding back to `df_tr_fold`, each row's own target value contributed to `enc[its_category]`. This is sometimes called train-fold self-leakage or self-inclusion bias.

For this dataset the practical impact is negligible:
- For `town` and `planning_area`, groups typically have hundreds to thousands of transactions. Each individual row contributes ~1/N to its own encoded value — at N=1000 that is 0.1% influence.
- Sigmoid smoothing (k=10) already shrinks rare groups toward the global mean, which further reduces the effect.
- The same pattern exists in spatial encoding: when building KD-tree encodings for training rows, each point's own price is included among its neighbours. At 500 m radius in Singapore, a point typically has 50–500 neighbours, so self-inclusion adds ~0.2–2% bias.

The **OOF RMSE is unaffected** — it measures validation performance, which is computed clean. The training-side bias causes a small degree of optimism in training fit but does not propagate into the validation metric. The correct fix (leave-one-out encoding) adds implementation complexity for negligible gain at this group size.

### Label Encoding — `flat_type`, `flat_model`, `mrt_name`, `pri_sch_name`, `sec_sch_name`

Label encoding assigns integer indices to each category. Because the target is not involved, this is computed globally (fitting on train + test combined to handle unseen categories) and is safe at all times. Used here because all three tree models handle ordinal integers natively.

---

## Spatial Encoding (KD-Tree)

### Why Geohash Was Dropped

**v12** introduced geohash spatial encoding: each transaction is assigned to a geohash cell (~150 m × 150 m at precision 7) and the smoothed mean price of all transactions in that cell is used as a feature.

The problem is the **cell boundary discontinuity**. Geohash divides space into discrete rectangles. Two transactions 10 m apart on opposite sides of a cell boundary receive completely different encodings even though their location-based prices should be nearly identical. Expanding to 9 adjacent cells softened but did not remove this discontinuity.

### Why KD-Tree (v13+)

**cKDTree** (`scipy.spatial`) replaces geohash with continuous radius-based aggregation:

1. Build a KD-tree from training-fold coordinates (Latitude, Longitude converted to metres).
2. For each query point, find **all training transactions within r metres**.
3. Compute the smoothed mean price (and mean price-per-sqm) of those neighbours.

This gives a **fully continuous price surface** — no grid cells, no boundaries. A transaction at any location smoothly interpolates the prices of nearby training transactions.

```python
# Built from training-fold coordinates only → leakage-free for val/test
tree = cKDTree(coords_train)

# For each query point, radius search in O(n log n)
neighbours = tree.query_ball_point(query_coords, r=500, workers=-1)
smoothed_mean = sigmoid_smooth(len(neighbours), mean(prices[neighbours]), global_mean)
```

Four spatial features are added per fold:

| Feature | Radius | What it captures |
| ------- | ------ | ---------------- |
| `spatial_500m_te` | 500 m | Block-level mean price |
| `spatial_2000m_te` | 2000 m | District-level mean price |
| `spatial_500m_psm` | 500 m | Block-level mean price-per-sqm (size-normalised) |
| `spatial_2000m_psm` | 2000 m | District-level mean price-per-sqm |

PSM (price per sqm) features add size-normalised context — a 3-room flat near expensive 5-room blocks benefits from a high neighbourhood mean even though its raw price is lower.

---

## Pipeline Versions

| Version | Key change | Ensemble OOF RMSE | Kaggle RMSE |
| ------- | ---------- | ----------------- | ----------- |
| v1 | Baseline (80/20 split, LGB/XGB/CAT) | 21,220 | 22,067 |
| v2 | 5-fold CV | 21,211 | 21,498 |
| v3 | PCA-informed: dropped 5 redundant cols | 21,195 | — |
| v4 | Target encoding for `town`/`planning_area` | 21,177 | — |
| v5 | Added CBD distance, price trend *(reverted)* | 21,195 | — |
| v6 | 5-fold OOF stacking | 21,542 | 21,466 |
| v7 | CatBoost Optuna HPO (50 trials) | 21,525 | — |
| v8 | CatBoost warm-start depth ≤ 12 — confirmed convergence | 21,525 | — |
| v9 | LightGBM + XGBoost Optuna HPO (50 trials each) | 21,459 | — |
| v10 | Dual output: 80/20 + 5-fold OOF | 5-fold: 21,456 | 5-fold: 21,413 |
| v11 | Leak-free target encoding (computed inside fold loop) | 21,461 | — |
| v12 | Geohash spatial encoding (4 features) | 21,680 | 21,714 |
| v13 | KD-tree replaces geohash | 21,456 | 21,396 |
| v14 | Dropped 29 low-importance features; 5-fold OOF only | 21,369 | 21,284 |
| v15 | Re-tuned all 3 models with 3-fold inner CV HPO | 21,334 | **21,243** |
| v16 | Time-aware spatial encoding (24-month window) | 21,364 (worse) | — |
| v17 | Postal-code target encoding (block-level) | 21,949 (worse) | — |
| v18 | Added 1000 m spatial radius | 21,334 (no gain) | — |
| v19 | Added `spatial_2000m_psm`; Re-tuned models with 3-fold inner CV HPO | 21,320 | — |
| **v20** | Dropped all distance cols (redundant with log-transform distance) | **21,325** | **21,225** |
| v21 | Dropped target encoding and changed spatial radius from 2000 to 1000m | 21,317 | 21,240 |

---

## Pros and Cons

### Pros

| Strength | Detail |
| -------- | ------ |
| **High accuracy** | R² = 0.978, MAPE = 3.47%, MAE ≈ SGD 15,300 |
| **Robust validation** | 5-fold OOF ensures every row is evaluated out-of-sample |
| **All models tuned** | Optuna TPE with 50 trials per model, 3-fold inner CV |
| **Leak-free val/test encoding** | Target and spatial encoding stats computed from training-fold data only |
| **Continuous spatial features** | KD-tree radius search provides a smooth price surface with no grid-boundary discontinuities |
| **Diverse ensemble** | Three different GBDT implementations; Nelder-Mead finds optimal weights |
| **Interpretable features** | All engineered features have clear domain meaning |

### Cons

| Limitation | Detail |
| ---------- | ------ |
| **Train-fold self-leakage** | Target encoding and spatial encoding include each training row's own price in its own feature value. Impact is small (1/N contribution per row) but technically present; leave-one-out encoding would fully eliminate it |
| **No macro market signal** | HDB Resale Price Index (RPI) not incorporated; `tranc_period` is a coarse proxy |
| **Ensemble diversity ceiling** | All three models are GBDT variants; a neural network (TabNet) or linear model would add more orthogonal signal |
| **Static model** | No retraining mechanism; predictions will drift as the market evolves |

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
│   ├── hdb_ml_pipeline_v20.py          ← current version (use this)
│   └── hdb_ml_pipeline_v1..v19.py      ← earlier iterations
└── submission/
    └── submission_v20_5fold.csv         ← latest (recommended)
```

## Running the Pipeline

```bash
cd script
python hdb_ml_pipeline_v20.py
```

**Requirements:** `pandas numpy scikit-learn lightgbm xgboost catboost scipy optuna`

**Runtime:** approximately 25–35 minutes on Apple M-series CPU (no HPO — params hardcoded from v19).

The script outputs `submission_v20_5fold.csv` and prints a full feature importance table (all features, sorted by mean gain across LGB / XGB / CAT) averaged across all 5 folds.
