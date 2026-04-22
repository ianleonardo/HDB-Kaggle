"""
HDB Resale Price Prediction Pipeline — LOG TARGET
Primary model : CatBoost
Secondary     : LightGBM, XGBoost (for comparison)
Target        : log(resale_price); predictions exponentiated back for RMSE
Split         : 80% train / 20% validation from train.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
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

# ─── Feature Engineering ──────────────────────────────────────────────────────
def feature_engineering(df):
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

    # --- Area features ---
    df['sqm_per_storey'] = df['floor_area_sqm'] / df['mid_storey'].replace(0, np.nan)

    # --- Distance-based features ---
    df['log_mrt_dist']    = np.log1p(df['mrt_nearest_distance'])
    df['log_mall_dist']   = np.log1p(df['Mall_Nearest_Distance'].fillna(df['Mall_Nearest_Distance'].median()))
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


train = feature_engineering(train)
test  = feature_engineering(test)

# ─── Categorical Encoding ─────────────────────────────────────────────────────
CAT_COLS = ['town', 'flat_type', 'flat_model', 'full_flat_type',
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
    'storey_range', 'postal', 'bus_stop_name'
] + CAT_COLS

FEATURES = [c for c in train.columns if c not in DROP_COLS]
print(f"\nNumber of features: {len(FEATURES)}")

X      = train[FEATURES]
X_test = test[FEATURES]

# ─── Log-transform the target ─────────────────────────────────────────────────
y_raw = train[TARGET]
y     = np.log(y_raw)      # train on log scale
print(f"\nTarget: log(resale_price)  |  mean={y.mean():.4f}, std={y.std():.4f}")

# ─── 80/20 Train/Validation Split ────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Keep original-scale val labels for RMSE reporting
y_val_raw = np.exp(y_val)

print(f"Train size: {len(X_tr):,}  |  Val size: {len(X_val):,}")

def rmse_orig(y_true_raw, log_preds):
    """Exponentiate log predictions then compute RMSE on original price scale."""
    return root_mean_squared_error(y_true_raw, np.exp(log_preds))

# ─── CatBoost (Primary) ──────────────────────────────────────────────────────
print("\n=== CatBoost [PRIMARY] ===")
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
    'verbose':             200,
    'random_seed':         42,
    'task_type':           'CPU',
}
cat_model = cb.CatBoostRegressor(**cat_params)
cat_model.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              early_stopping_rounds=100,
              use_best_model=True)
val_cat_log = cat_model.predict(X_val)
cat_rmse    = rmse_orig(y_val_raw, val_cat_log)
print(f"  CatBoost Val RMSE (original scale): {cat_rmse:,.0f}")

# ─── LightGBM (Comparison) ───────────────────────────────────────────────────
print("\n=== LightGBM [comparison] ===")
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
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(200)])
val_lgb_log = lgb_model.predict(X_val)
lgb_rmse    = rmse_orig(y_val_raw, val_lgb_log)
print(f"  LightGBM Val RMSE (original scale): {lgb_rmse:,.0f}")

# ─── XGBoost (Comparison) ────────────────────────────────────────────────────
print("\n=== XGBoost [comparison] ===")
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
xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100, verbosity=0)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
val_xgb_log = xgb_model.predict(X_val)
xgb_rmse    = rmse_orig(y_val_raw, val_xgb_log)
print(f"  XGBoost Val RMSE (original scale): {xgb_rmse:,.0f}")

# ─── Ensemble (Optimised Weighted Average on log-scale val preds) ─────────────
from scipy.optimize import minimize

def ensemble_rmse(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend_log = w[0]*val_cat_log + w[1]*val_lgb_log + w[2]*val_xgb_log
    return rmse_orig(y_val_raw, blend_log)

res    = minimize(ensemble_rmse, [1, 1, 1], method='Nelder-Mead',
                  options={'maxiter': 1000, 'xatol': 1e-6})
best_w = res.x / res.x.sum()

val_blend_log = best_w[0]*val_cat_log + best_w[1]*val_lgb_log + best_w[2]*val_xgb_log
blend_rmse    = rmse_orig(y_val_raw, val_blend_log)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Validation RMSE Summary — LOG target (20% hold-out) ===")
print(f"  CatBoost [PRIMARY] : {cat_rmse:,.0f}")
print(f"  LightGBM           : {lgb_rmse:,.0f}")
print(f"  XGBoost            : {xgb_rmse:,.0f}")
print(f"  Ensemble           : {blend_rmse:,.0f}  "
      f"(CAT={best_w[0]:.3f}, LGB={best_w[1]:.3f}, XGB={best_w[2]:.3f})")

# ─── Predict on test.csv using ensemble ──────────────────────────────────────
test_blend_log = (best_w[0] * cat_model.predict(X_test) +
                  best_w[1] * lgb_model.predict(X_test) +
                  best_w[2] * xgb_model.predict(X_test))
preds_final = np.exp(test_blend_log)   # back to original price scale

# ─── Submission ───────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": test[ID_COL], "Predicted": preds_final})
sub.to_csv('../submission/submission_ensemble_log.csv', index=False)
print("\nSubmission saved to submission/submission_ensemble_log.csv")
print(sub.head())
