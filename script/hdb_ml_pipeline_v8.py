"""
HDB Resale Price Prediction Pipeline  —  v8
Primary model : CatBoost (Optuna warm-start, extended search)
Secondary     : LightGBM, XGBoost
Metric        : RMSE
Validation    : 5-fold KFold OOF stacking

Changes vs v7:
  - Optuna warm-start from v7 best trial (trial #36, HPO RMSE=21,760)
    Seed params enqueued as trial 0 so TPE explores around the known optimum.
  - depth upper bound expanded 10 → 12
    v7 best depth=10 hit the boundary — more capacity may still help.
  - 20 trials only (focused search around known-good region, not full 50)
  - Everything else identical to v7 (features, OOF, LGB/XGB params)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── GPU toggle ───────────────────────────────────────────────────────────────
USE_GPU = False  # set True if NVIDIA GPU + CatBoost GPU libraries are available

CAT_TASK_TYPE = 'GPU' if USE_GPU else 'CPU'
XGB_DEVICE    = 'cuda' if USE_GPU else None

print(f"GPU mode: {USE_GPU}  (CatBoost={CAT_TASK_TYPE})")

# ─── v7 best params — used as warm-start seed for Optuna ─────────────────────
V7_BEST_PARAMS = {
    'learning_rate':       0.0318637615066506,
    'depth':               10,
    'l2_leaf_reg':         1.2695026389840243,
    'random_strength':     1.540606655478695,
    'bagging_temperature': 1.2874233422351384,
    'border_count':        205,
}

# ─── Load Data ────────────────────────────────────────────────────────────────
train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv',  low_memory=False)

print(f"Train: {train.shape}, Test: {test.shape}")

TARGET = 'resale_price'
ID_COL = 'id'

# ─── Metrics helper ───────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2

def print_metrics(label, y_true, y_pred):
    rmse, mae, mape, r2 = metrics(y_true, y_pred)
    print(f"  {label:<40}  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:>6.2f}%  R²={r2:.4f}")

# ─── Feature Engineering ──────────────────────────────────────────────────────
def feature_engineering(df, mall_dist_median):
    df = df.copy()

    # --- Time features ---
    df['lease_remaining_years'] = 99 - (df['Tranc_Year'] - df['lease_commence_date'])
    df['lease_remaining_pct']   = df['lease_remaining_years'] / 99.0
    df['month_sin']    = np.sin(2 * np.pi * df['Tranc_Month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['Tranc_Month'] / 12)
    df['tranc_period'] = df['Tranc_Year'] * 12 + df['Tranc_Month']

    # --- Floor / Storey features ---
    df['storey_ratio']  = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan)
    df['is_high_floor'] = (df['mid_storey'] >= 20).astype(int)
    df['floor_band']    = pd.cut(df['mid_storey'],
                                 bins=[0, 5, 10, 15, 20, 30, 50, 999],
                                 labels=[1, 2, 3, 4, 5, 6, 7]).astype(float)

    # --- Distance-based features ---
    df['log_mrt_dist']    = np.log1p(df['mrt_nearest_distance'])
    df['log_mall_dist']   = np.log1p(df['Mall_Nearest_Distance'].fillna(mall_dist_median))
    df['log_hawker_dist'] = np.log1p(df['Hawker_Nearest_Distance'])
    df['log_bus_dist']    = np.log1p(df['bus_stop_nearest_distance'])
    df['log_pri_dist']    = np.log1p(df['pri_sch_nearest_distance'])
    df['log_sec_dist']    = np.log1p(df['sec_sch_nearest_dist'])

    df['accessibility_score'] = (
        df['log_mrt_dist'] * 0.4 +
        df['log_mall_dist'] * 0.2 +
        df['log_hawker_dist'] * 0.2 +
        df['log_bus_dist'] * 0.2
    )

    for col in ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']:
        df[col] = df[col].fillna(0)

    # --- Building / Block features ---
    df['is_mrt_interchange'] = df['mrt_interchange'].fillna(0).astype(int)
    df['is_bus_interchange'] = df['bus_interchange'].fillna(0).astype(int)
    df['total_sold_units']   = (
        df[['1room_sold','2room_sold','3room_sold','4room_sold',
            '5room_sold','exec_sold','multigen_sold','studio_apartment_sold']].sum(axis=1)
    )
    df['total_rental_units'] = df[['1room_rental','2room_rental','3room_rental','other_room_rental']].sum(axis=1)
    df['sold_ratio']         = df['total_sold_units'] / df['total_dwelling_units'].replace(0, np.nan)

    # --- School quality ---
    df['school_quality']     = df['cutoff_point'].fillna(0) + df['affiliation'].fillna(0) * 10
    df['pri_school_quality'] = df['pri_sch_affiliation'].fillna(0) * 10 + (1 / (df['pri_sch_nearest_distance'] + 1))

    # --- Y/N flag cols → binary ---
    for col in ['residential', 'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion']:
        df[col] = (df[col] == 'Y').astype(int)

    # --- Interaction features ---
    df['area_x_storey']               = df['floor_area_sqm'] * df['mid_storey']
    df['area_x_lease_rem']            = df['floor_area_sqm'] * df['lease_remaining_years']
    df['storey_x_lease_rem']          = df['mid_storey'] * df['lease_remaining_years']
    df['year_completed_x_floor_area'] = df['year_completed'] * df['floor_area_sqm']

    return df


mall_dist_median = train['Mall_Nearest_Distance'].median()
train = feature_engineering(train, mall_dist_median)
test  = feature_engineering(test,  mall_dist_median)

# ─── Target Encoding for town / planning_area ─────────────────────────────────
def smoothed_target_encode(df_train, df_test, col, target, smoothing_k=10):
    global_mean = df_train[target].mean()
    agg = df_train.groupby(col)[target].agg(['mean', 'count'])
    smooth = 1.0 / (1.0 + np.exp(-(agg['count'] - smoothing_k) / smoothing_k))
    encoded = global_mean * (1 - smooth) + agg['mean'] * smooth
    train_enc = df_train[col].map(encoded).fillna(global_mean)
    test_enc  = df_test[col].map(encoded).fillna(global_mean)
    return train_enc, test_enc

TARGET_ENC_COLS = ['town', 'planning_area']
for col in TARGET_ENC_COLS:
    train[col + '_te'], test[col + '_te'] = smoothed_target_encode(
        train, test, col, TARGET
    )
    print(f"Target encoded '{col}': {train[col].nunique()} levels  "
          f"| range [{train[col+'_te'].min():.0f}, {train[col+'_te'].max():.0f}]")

# ─── Label Encoding (remaining categoricals) ─────────────────────────────────
LABEL_ENC_COLS = ['flat_type', 'flat_model', 'mrt_name', 'pri_sch_name', 'sec_sch_name']

le_dict = {}
for col in LABEL_ENC_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
    le.fit(combined)
    train[col + '_enc'] = le.transform(train[col].astype(str))
    test[col + '_enc']  = le.transform(test[col].astype(str))
    le_dict[col] = le

# ─── Select Features ──────────────────────────────────────────────────────────
ALL_CAT_ORIGINALS = TARGET_ENC_COLS + LABEL_ENC_COLS
REDUNDANT_COLS    = ['floor_area_sqft', 'hdb_age', 'lower', 'upper', 'mid']

DROP_COLS = [
    TARGET, ID_COL,
    'Tranc_YearMonth', 'block', 'street_name', 'address',
    'storey_range', 'postal', 'bus_stop_name', 'full_flat_type',
] + ALL_CAT_ORIGINALS + REDUNDANT_COLS

FEATURES = [c for c in train.columns if c not in DROP_COLS]
print(f"\nNumber of features: {len(FEATURES)}")

X      = train[FEATURES].reset_index(drop=True)
y      = train[TARGET].reset_index(drop=True)
X_test = test[FEATURES]

# ─── Optuna HPO — CatBoost (warm-start + extended depth) ─────────────────────
print("\n" + "="*65)
print("Phase 1: Optuna HPO — 20 trials (warm-start from v7, depth≤12)")
print("="*65)

X_hpo_tr, X_hpo_val, y_hpo_tr, y_hpo_val = train_test_split(
    X, y, test_size=0.20, random_state=0
)

def cat_objective(trial):
    params = {
        'loss_function':       'RMSE',
        'iterations':          3000,
        'od_type':             'Iter',
        'od_wait':             100,
        'verbose':             0,
        'random_seed':         42,
        'task_type':           CAT_TASK_TYPE,
        'learning_rate':       trial.suggest_float('learning_rate',       0.01,  0.15,  log=True),
        'depth':               trial.suggest_int  ('depth',               4,     12),   # ← extended to 12
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg',         1.0,   30.0,  log=True),
        'random_strength':     trial.suggest_float('random_strength',     0.1,   5.0,   log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0,   2.0),
        'border_count':        trial.suggest_int  ('border_count',        32,    255),
    }
    m = cb.CatBoostRegressor(**params)
    m.fit(X_hpo_tr, y_hpo_tr,
          eval_set=(X_hpo_val, y_hpo_val),
          early_stopping_rounds=100,
          verbose=False)
    return root_mean_squared_error(y_hpo_val, m.predict(X_hpo_val))

study = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=42))

# Warm-start: enqueue v7 best params as trial 0 so TPE builds its model
# around the known-good region immediately rather than exploring blindly.
study.enqueue_trial(V7_BEST_PARAMS)

study.optimize(cat_objective, n_trials=20, show_progress_bar=True)

best_cat_params = {
    'loss_function':       'RMSE',
    'iterations':          3000,
    'od_type':             'Iter',
    'od_wait':             100,
    'verbose':             0,
    'random_seed':         42,
    'task_type':           CAT_TASK_TYPE,
    **study.best_params,
}

# Compare against v7 baseline
v7_rmse   = 21_760
v8_rmse   = study.best_value
improved  = v7_rmse - v8_rmse

print(f"\nv7 HPO RMSE : {v7_rmse:,.0f}")
print(f"v8 HPO RMSE : {v8_rmse:,.0f}  ({'↓ improved by ' + f'{improved:,.0f}' if improved > 0 else '↑ no improvement — v7 params still best'})")
print(f"\nBest CatBoost params (trial #{study.best_trial.number}):")
for k, v in study.best_params.items():
    marker = ' ◄ changed' if abs(v - V7_BEST_PARAMS.get(k, v)) / (abs(V7_BEST_PARAMS.get(k, v)) + 1e-9) > 0.05 else ''
    print(f"  {k:<25} = {v}{marker}")

# ─── LGB / XGB base params ────────────────────────────────────────────────────
lgb_params = {
    'objective':         'regression',
    'metric':            'rmse',
    'n_estimators':      3000,
    'learning_rate':     0.05,
    'num_leaves':        255,
    'max_depth':         -1,
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      1,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      42,
}

xgb_params = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'n_estimators':     3000,
    'learning_rate':    0.05,
    'max_depth':        8,
    'min_child_weight': 5,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'tree_method':      'hist',
    **({'device': XGB_DEVICE} if XGB_DEVICE else {}),
    'random_state':     42,
    'n_jobs':           -1,
}

# ─── 5-Fold OOF Training ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("Phase 2: 5-fold OOF with best CatBoost params")
print("="*65)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n{'─'*52}")
    print(f"  Fold {fold} / {N_FOLDS}   train={len(tr_idx):,}   val={len(val_idx):,}")
    print(f"{'─'*52}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # LightGBM
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
    oof_lgb[val_idx]  = m_lgb.predict(X_val)
    pred_lgb         += m_lgb.predict(X_test) / N_FOLDS
    print_metrics(f"  LGB fold {fold}", y_val, oof_lgb[val_idx])

    # XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100, verbosity=0)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx]  = m_xgb.predict(X_val)
    pred_xgb         += m_xgb.predict(X_test) / N_FOLDS
    print_metrics(f"  XGB fold {fold}", y_val, oof_xgb[val_idx])

    # CatBoost (warm-started, extended depth)
    m_cat = cb.CatBoostRegressor(**best_cat_params)
    m_cat.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              early_stopping_rounds=100,
              use_best_model=True,
              verbose=False)
    oof_cat[val_idx]  = m_cat.predict(X_val)
    pred_cat         += m_cat.predict(X_test) / N_FOLDS
    print_metrics(f"  CAT fold {fold}", y_val, oof_cat[val_idx])

# ─── Ensemble weights on full OOF ────────────────────────────────────────────
def oof_ensemble_rmse(weights):
    w = np.array(weights)
    w = w / w.sum()
    return root_mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat)

res    = minimize(oof_ensemble_rmse, [1, 1, 1], method='Nelder-Mead',
                  options={'maxiter': 1000, 'xatol': 1e-6})
best_w = res.x / res.x.sum()

oof_blend = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cat

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"=== OOF Metrics — v8 vs v7 ===")
print(f"{'─'*65}")
print_metrics("LightGBM  (OOF)",          y, oof_lgb)
print_metrics("XGBoost   (OOF)",          y, oof_xgb)
print_metrics("CatBoost  (OOF, tuned)",   y, oof_cat)
print_metrics(
    f"Ensemble  (C={best_w[2]:.2f} L={best_w[0]:.2f} X={best_w[1]:.2f})",
    y, oof_blend
)
print(f"{'─'*65}")
print(f"  v7 ensemble OOF RMSE : 21,525")
_, _, _, _ = metrics(y, oof_blend)
delta = 21_525 - root_mean_squared_error(y, oof_blend)
print(f"  v8 ensemble OOF RMSE : {root_mean_squared_error(y, oof_blend):,.0f}  "
      f"({'↓ ' + f'{delta:,.0f} improvement' if delta > 0 else '↑ ' + f'{abs(delta):,.0f} regression'})")
print(f"{'='*65}")

# ─── Predict on test.csv ─────────────────────────────────────────────────────
preds_final = best_w[0]*pred_lgb + best_w[1]*pred_xgb + best_w[2]*pred_cat

# ─── Submission ───────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": test[ID_COL], "Predicted": preds_final})
sub.to_csv('../submission/submission_ensemble_v8.csv', index=False)
print("\nSubmission saved to submission/submission_ensemble_v8.csv")
print(sub.head())
