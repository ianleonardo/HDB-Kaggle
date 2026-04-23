"""
HDB Resale Price Prediction Pipeline  —  v9
All three models Optuna-tuned
Metric        : RMSE
Validation    : 5-fold KFold OOF stacking

Changes vs v8:
  - Optuna HPO added for LightGBM (50 trials) and XGBoost (50 trials)
    Previously LGB/XGB used hand-picked defaults; they contribute 44% of
    ensemble weight (L=0.24 X=0.20) so tuning them has real upside.
  - CatBoost reuses v7/v8 confirmed best params (no re-tuning needed —
    v8 warm-start confirmed convergence).
  - Three-phase structure:
      Phase 1a  LightGBM  HPO  50 trials
      Phase 1b  XGBoost   HPO  50 trials
      Phase 2   5-fold OOF with all three tuned models
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
USE_GPU = False  # Apple M-series: Metal GPU not supported by CatBoost/XGBoost

CAT_TASK_TYPE = 'GPU' if USE_GPU else 'CPU'
XGB_DEVICE    = 'cuda' if USE_GPU else None

# ─── Confirmed best CatBoost params from v7/v8 ───────────────────────────────
BEST_CAT_PARAMS = {
    'loss_function':       'RMSE',
    'iterations':          3000,
    'od_type':             'Iter',
    'od_wait':             100,
    'verbose':             0,
    'random_seed':         42,
    'task_type':           CAT_TASK_TYPE,
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

# ─── Shared HPO split (same seed for fair comparison across models) ───────────
X_hpo_tr, X_hpo_val, y_hpo_tr, y_hpo_val = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ─── Phase 1a: Optuna HPO — LightGBM ─────────────────────────────────────────
print("\n" + "="*65)
print("Phase 1a: Optuna HPO — LightGBM (50 trials)")
print("="*65)

def lgb_objective(trial):
    params = {
        'objective':         'regression',
        'metric':            'rmse',
        'n_estimators':      3000,
        'verbose':           -1,
        'n_jobs':            -1,
        'random_state':      42,
        'bagging_freq':      1,
        'learning_rate':     trial.suggest_float('learning_rate',     0.01,  0.15,  log=True),
        'num_leaves':        trial.suggest_int  ('num_leaves',        50,    500),
        'min_child_samples': trial.suggest_int  ('min_child_samples', 5,     100),
        'feature_fraction':  trial.suggest_float('feature_fraction',  0.5,   1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.5,   1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',         1e-4,  5.0,   log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda',        1e-4,  5.0,   log=True),
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_hpo_tr, y_hpo_tr,
          eval_set=[(X_hpo_val, y_hpo_val)],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(0)])
    return root_mean_squared_error(y_hpo_val, m.predict(X_hpo_val))

lgb_study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)

best_lgb_params = {
    'objective':    'regression',
    'metric':       'rmse',
    'n_estimators': 3000,
    'verbose':      -1,
    'n_jobs':       -1,
    'random_state': 42,
    'bagging_freq': 1,
    **lgb_study.best_params,
}

print(f"\nBest LightGBM (trial #{lgb_study.best_trial.number}, "
      f"HPO val RMSE={lgb_study.best_value:,.0f}):")
for k, v in lgb_study.best_params.items():
    print(f"  {k:<25} = {v}")

# ─── Phase 1b: Optuna HPO — XGBoost ──────────────────────────────────────────
print("\n" + "="*65)
print("Phase 1b: Optuna HPO — XGBoost (50 trials)")
print("="*65)

def xgb_objective(trial):
    params = {
        'objective':        'reg:squarederror',
        'eval_metric':      'rmse',
        'n_estimators':     3000,
        'tree_method':      'hist',
        'random_state':     42,
        'n_jobs':           -1,
        **({'device': XGB_DEVICE} if XGB_DEVICE else {}),
        'learning_rate':     trial.suggest_float('learning_rate',     0.01,  0.15,  log=True),
        'max_depth':         trial.suggest_int  ('max_depth',         4,     12),
        'min_child_weight':  trial.suggest_int  ('min_child_weight',  1,     20),
        'subsample':         trial.suggest_float('subsample',         0.5,   1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree',  0.5,   1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',         1e-4,  5.0,   log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda',        1e-4,  5.0,   log=True),
        'gamma':             trial.suggest_float('gamma',             0.0,   2.0),
    }
    m = xgb.XGBRegressor(**params, early_stopping_rounds=100, verbosity=0)
    m.fit(X_hpo_tr, y_hpo_tr,
          eval_set=[(X_hpo_val, y_hpo_val)],
          verbose=False)
    return root_mean_squared_error(y_hpo_val, m.predict(X_hpo_val))

xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

best_xgb_params = {
    'objective':    'reg:squarederror',
    'eval_metric':  'rmse',
    'n_estimators': 3000,
    'tree_method':  'hist',
    'random_state': 42,
    'n_jobs':       -1,
    **({'device': XGB_DEVICE} if XGB_DEVICE else {}),
    **xgb_study.best_params,
}

print(f"\nBest XGBoost (trial #{xgb_study.best_trial.number}, "
      f"HPO val RMSE={xgb_study.best_value:,.0f}):")
for k, v in xgb_study.best_params.items():
    print(f"  {k:<25} = {v}")

# ─── HPO Summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  HPO Summary (val split RMSE):")
print(f"  LightGBM  default≈22,000  →  tuned={lgb_study.best_value:,.0f}")
print(f"  XGBoost   default≈22,000  →  tuned={xgb_study.best_value:,.0f}")
print(f"  CatBoost  (v7/v8 confirmed best)  HPO val=21,760")
print(f"{'─'*65}")

# ─── Phase 2: 5-Fold OOF with all three tuned models ─────────────────────────
print("\n" + "="*65)
print("Phase 2: 5-fold OOF — all three models tuned")
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

    # LightGBM (tuned)
    m_lgb = lgb.LGBMRegressor(**best_lgb_params)
    m_lgb.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
    oof_lgb[val_idx]  = m_lgb.predict(X_val)
    pred_lgb         += m_lgb.predict(X_test) / N_FOLDS
    print_metrics(f"  LGB fold {fold}", y_val, oof_lgb[val_idx])

    # XGBoost (tuned)
    m_xgb = xgb.XGBRegressor(**best_xgb_params, early_stopping_rounds=100, verbosity=0)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx]  = m_xgb.predict(X_val)
    pred_xgb         += m_xgb.predict(X_test) / N_FOLDS
    print_metrics(f"  XGB fold {fold}", y_val, oof_xgb[val_idx])

    # CatBoost (v7/v8 confirmed best params)
    m_cat = cb.CatBoostRegressor(**BEST_CAT_PARAMS)
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

# ─── Final Summary ────────────────────────────────────────────────────────────
v8_ensemble_rmse = 21_525
v9_ensemble_rmse = root_mean_squared_error(y, oof_blend)
delta            = v8_ensemble_rmse - v9_ensemble_rmse

print(f"\n{'='*65}")
print(f"=== OOF Metrics — v9 (all models tuned) ===")
print(f"{'─'*65}")
print_metrics("LightGBM  (OOF, tuned)",     y, oof_lgb)
print_metrics("XGBoost   (OOF, tuned)",     y, oof_xgb)
print_metrics("CatBoost  (OOF, tuned)",     y, oof_cat)
print_metrics(
    f"Ensemble  (C={best_w[2]:.2f} L={best_w[0]:.2f} X={best_w[1]:.2f})",
    y, oof_blend
)
print(f"{'─'*65}")
print(f"  v8 ensemble OOF RMSE : {v8_ensemble_rmse:,}")
print(f"  v9 ensemble OOF RMSE : {v9_ensemble_rmse:,.0f}  "
      f"({'↓ improved by ' + f'{delta:,.0f}' if delta > 0 else '↑ regressed by ' + f'{abs(delta):,.0f}'})")
print(f"{'='*65}")

# ─── Predict & Submit ─────────────────────────────────────────────────────────
preds_final = best_w[0]*pred_lgb + best_w[1]*pred_xgb + best_w[2]*pred_cat

sub = pd.DataFrame({"Id": test[ID_COL], "Predicted": preds_final})
sub.to_csv('../submission/submission_ensemble_v9.csv', index=False)
print("\nSubmission saved to submission/submission_ensemble_v9.csv")
print(sub.head())
