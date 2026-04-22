"""
HDB Resale Price Prediction Pipeline
Primary model : CatBoost
Secondary     : LightGBM, XGBoost (for comparison)
Metric        : RMSE
Validation    : 5-fold CV (OOF predictions); ensemble weights optimised on OOF

Fixes applied vs v1:
  #3  mall_dist_median computed from train only, passed into feature_engineering
  #4  sqm_per_storey dropped (floor_area / mid_storey has no real meaning)
  #5  ensemble weights optimised on OOF, not the same val set
  #10 5-fold CV replaces single 80/20 split
  #11 full_flat_type dropped (exact concat of flat_type + flat_model, zero unique signal)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
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
N_FOLDS = 5

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
    # sqm_per_storey removed: floor_area / mid_storey has no physical meaning

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
    df['area_x_storey']      = df['floor_area_sqm'] * df['mid_storey']
    df['area_x_lease_rem']   = df['floor_area_sqm'] * df['lease_remaining_years']
    df['storey_x_lease_rem'] = df['mid_storey'] * df['lease_remaining_years']

    return df


# Compute imputation stats from train only, then apply to both
mall_dist_median = train['Mall_Nearest_Distance'].median()

train = feature_engineering(train, mall_dist_median)
test  = feature_engineering(test,  mall_dist_median)

# ─── Categorical Encoding ─────────────────────────────────────────────────────
# full_flat_type dropped: it is exactly flat_type + " " + flat_model (confirmed),
# so it duplicates information already captured by those two columns.
CAT_COLS = ['town', 'flat_type', 'flat_model',
            'planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name']

le_dict = {}
for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
    le.fit(combined)
    train[col + '_enc'] = le.transform(train[col].astype(str))
    test[col + '_enc']  = le.transform(test[col].astype(str))
    le_dict[col] = le

# ─── Select Features ──────────────────────────────────────────────────────────
DROP_COLS = [
    TARGET, ID_COL,
    'Tranc_YearMonth', 'block', 'street_name', 'address',
    'storey_range', 'postal', 'bus_stop_name', 'full_flat_type',
] + CAT_COLS

FEATURES = [c for c in train.columns if c not in DROP_COLS]
print(f"\nNumber of features: {len(FEATURES)}")

X      = train[FEATURES]
y      = train[TARGET]
X_test = test[FEATURES]

# ─── 5-Fold CV ───────────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))

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

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    print(f"\n── Fold {fold + 1}/{N_FOLDS} ──────────────────────────────")

    # LightGBM
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
    oof_lgb[val_idx]  = m_lgb.predict(X_val)
    preds_lgb        += m_lgb.predict(X_test) / N_FOLDS
    rmse, mae, mape, r2 = metrics(y_val, oof_lgb[val_idx])
    print(f"  LGB  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:.2f}%  R²={r2:.4f}")

    # XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100, verbosity=0)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx]  = m_xgb.predict(X_val)
    preds_xgb        += m_xgb.predict(X_test) / N_FOLDS
    rmse, mae, mape, r2 = metrics(y_val, oof_xgb[val_idx])
    print(f"  XGB  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:.2f}%  R²={r2:.4f}")

    # CatBoost
    m_cat = cb.CatBoostRegressor(**cat_params)
    m_cat.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              early_stopping_rounds=100,
              use_best_model=True,
              verbose=False)
    oof_cat[val_idx]  = m_cat.predict(X_val)
    preds_cat        += m_cat.predict(X_test) / N_FOLDS
    rmse, mae, mape, r2 = metrics(y_val, oof_cat[val_idx])
    print(f"  CAT  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:.2f}%  R²={r2:.4f}")

lgb_oof_rmse = root_mean_squared_error(y, oof_lgb)
xgb_oof_rmse = root_mean_squared_error(y, oof_xgb)
cat_oof_rmse = root_mean_squared_error(y, oof_cat)

# ─── Ensemble weights on OOF (no leakage — each row was held-out) ─────────────
def oof_ensemble_rmse(weights):
    w = np.array(weights)
    w = w / w.sum()
    return root_mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat)

res    = minimize(oof_ensemble_rmse, [1, 1, 1], method='Nelder-Mead',
                  options={'maxiter': 1000, 'xatol': 1e-6})
best_w = res.x / res.x.sum()

oof_blend  = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cat

# ─── Metrics helper ──────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2

def print_metrics(label, y_true, y_pred):
    rmse, mae, mape, r2 = metrics(y_true, y_pred)
    print(f"  {label:<20}  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:>6.2f}%  R²={r2:.4f}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n=== OOF Metrics Summary ({N_FOLDS}-fold CV) ===")
print_metrics("LightGBM",           y, oof_lgb)
print_metrics("XGBoost",            y, oof_xgb)
print_metrics("CatBoost [PRIMARY]", y, oof_cat)
print_metrics(f"Ensemble (CAT={best_w[2]:.2f} LGB={best_w[0]:.2f} XGB={best_w[1]:.2f})",
              y, oof_blend)

# ─── Predict on test.csv ─────────────────────────────────────────────────────
preds_final = best_w[0]*preds_lgb + best_w[1]*preds_xgb + best_w[2]*preds_cat

# ─── Submission ───────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": test[ID_COL], "Predicted": preds_final})
sub.to_csv('../submission/submission_ensemble.csv', index=False)
print("\nSubmission saved to submission/submission_ensemble.csv")
print(sub.head())
