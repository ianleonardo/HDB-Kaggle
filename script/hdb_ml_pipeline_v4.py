"""
HDB Resale Price Prediction Pipeline  —  v4
Primary model : CatBoost
Secondary     : LightGBM, XGBoost, Ridge (for comparison / ensemble)
Metric        : RMSE
Validation    : Single 80/20 random split; ensemble weights optimised on the
                held-out 20 % validation set.

Changes vs v3 (PCA-informed cleanup):
  - Target encoding (smoothed mean) for 'town' and 'planning_area'
      replaces label encoding for those two columns.
      Rationale: label encoding assigns arbitrary ordinal ranks to nominal
      categories; target encoding gives each level a price-signal directly.
      Smoothing formula: encoded = global_mean*(1-s) + group_mean*s
      where s = sigmoid(count - k), k=10 — shrinks small groups toward global mean.
  - Ridge regression added as 4th ensemble member
      Trained on (imputed + scaled) features → captures linear price signal
      that tree splits can miss (e.g., strict monotone relationships).
      A sklearn Pipeline (SimpleImputer → StandardScaler → Ridge) handles
      NaN values and scale sensitivity cleanly.
  - 4-way Nelder-Mead ensemble: [LGB, XGB, CAT, Ridge]
  - Submission saved to submission_ensemble_v4.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

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
    print(f"  {label:<28}  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:>6.2f}%  R²={r2:.4f}")

# ─── Feature Engineering ──────────────────────────────────────────────────────
def feature_engineering(df, mall_dist_median):
    """
    mall_dist_median : median of Mall_Nearest_Distance from train only.
                       Pass the same value for both train and test to avoid leakage.
    """
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
# Smoothed mean encoding: shrinks small groups toward the global mean.
# Stats computed on train only; applied to test via map (unseen → global mean).
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

X      = train[FEATURES]
y      = train[TARGET]
X_test = test[FEATURES]

# ─── 80 / 20 Random Split ────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain split : {len(X_tr):,} rows  |  Val split : {len(X_val):,} rows")

# ─── Model parameters ────────────────────────────────────────────────────────
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
    'random_state':     42,
    'n_jobs':           -1,
}

cat_params = {
    'loss_function':       'RMSE',
    'iterations':          3000,
    'learning_rate':       0.05,
    'depth':               8,
    'l2_leaf_reg':         3,
    'random_strength':     1,
    'bagging_temperature': 1,
    'od_type':             'Iter',
    'od_wait':             100,
    'verbose':             0,
    'random_seed':         42,
    'task_type':           'CPU',
}

# ─── Train models ─────────────────────────────────────────────────────────────
print("\n── Training LightGBM ───────────────────────────────")
m_lgb = lgb.LGBMRegressor(**lgb_params)
m_lgb.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(0)])
val_lgb  = m_lgb.predict(X_val)
pred_lgb = m_lgb.predict(X_test)
print_metrics("LightGBM", y_val, val_lgb)

print("\n── Training XGBoost ────────────────────────────────")
m_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100, verbosity=0)
m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
val_xgb  = m_xgb.predict(X_val)
pred_xgb = m_xgb.predict(X_test)
print_metrics("XGBoost", y_val, val_xgb)

print("\n── Training CatBoost ───────────────────────────────")
m_cat = cb.CatBoostRegressor(**cat_params)
m_cat.fit(X_tr, y_tr,
          eval_set=(X_val, y_val),
          early_stopping_rounds=100,
          use_best_model=True,
          verbose=False)
val_cat  = m_cat.predict(X_val)
pred_cat = m_cat.predict(X_test)
print_metrics("CatBoost [PRIMARY]", y_val, val_cat)

# ─── Ridge on residuals ───────────────────────────────────────────────────────
# Ridge captures linear price signal (e.g. strict monotone relationships) that
# tree splits can approximate only with many leaves.
# Pipeline handles NaN imputation and feature scaling internally so the same
# FEATURES list works for both tree models and Ridge without extra preprocessing.
print("\n── Training Ridge (residual correction) ────────────")
m_ridge = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('ridge',   Ridge(alpha=500)),
])
m_ridge.fit(X_tr, y_tr)
val_ridge  = m_ridge.predict(X_val)
pred_ridge = m_ridge.predict(X_test)
print_metrics("Ridge", y_val, val_ridge)

# ─── 4-way ensemble weights on validation set ────────────────────────────────
def val_ensemble_rmse(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = (w[0]*val_lgb + w[1]*val_xgb + w[2]*val_cat + w[3]*val_ridge)
    return root_mean_squared_error(y_val, blend)

res    = minimize(val_ensemble_rmse, [1, 1, 1, 0.2], method='Nelder-Mead',
                  options={'maxiter': 2000, 'xatol': 1e-6})
best_w = res.x / res.x.sum()

val_blend = (best_w[0]*val_lgb + best_w[1]*val_xgb +
             best_w[2]*val_cat + best_w[3]*val_ridge)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n=== Validation Metrics (80/20 split) ===")
print_metrics("LightGBM",           y_val, val_lgb)
print_metrics("XGBoost",            y_val, val_xgb)
print_metrics("CatBoost [PRIMARY]", y_val, val_cat)
print_metrics("Ridge",              y_val, val_ridge)
print_metrics(
    f"Ensemble (C={best_w[2]:.2f} L={best_w[0]:.2f} X={best_w[1]:.2f} R={best_w[3]:.2f})",
    y_val, val_blend
)

# ─── Predict on test.csv ─────────────────────────────────────────────────────
preds_final = (best_w[0]*pred_lgb + best_w[1]*pred_xgb +
               best_w[2]*pred_cat + best_w[3]*pred_ridge)

# ─── Submission ───────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": test[ID_COL], "Predicted": preds_final})
sub.to_csv('../submission/submission_ensemble_v4.csv', index=False)
print("\nSubmission saved to submission/submission_ensemble_v4.csv")
print(sub.head())
