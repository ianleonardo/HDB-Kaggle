"""
HDB Resale Price Prediction Pipeline
Primary model : CatBoost
Secondary     : LightGBM, XGBoost (for comparison)
Metric        : RMSE
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
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['Tranc_Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Tranc_Month'] / 12)
    # Transaction date as numeric (months since epoch)
    df['tranc_period'] = df['Tranc_Year'] * 12 + df['Tranc_Month']

    # --- Floor / Storey features ---
    df['storey_ratio'] = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan)
    df['is_high_floor'] = (df['mid_storey'] >= 20).astype(int)
    df['floor_band'] = pd.cut(df['mid_storey'],
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

    # Composite accessibility score (lower is better)
    df['accessibility_score'] = (
        df['log_mrt_dist'] * 0.4 +
        df['log_mall_dist'] * 0.2 +
        df['log_hawker_dist'] * 0.2 +
        df['log_bus_dist'] * 0.2
    )

    # Nearby amenity counts (fill NaN with 0 where binary)
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
    df['school_quality'] = df['cutoff_point'].fillna(0) + df['affiliation'].fillna(0) * 10
    df['pri_school_quality'] = df['pri_sch_affiliation'].fillna(0) * 10 + (1 / (df['pri_sch_nearest_distance'] + 1))

    # --- Y/N flag cols → binary ---
    for col in ['residential', 'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion']:
        df[col] = (df[col] == 'Y').astype(int)

    # --- Interaction features ---
    df['area_x_storey']     = df['floor_area_sqm'] * df['mid_storey']
    df['area_x_lease_rem']  = df['floor_area_sqm'] * df['lease_remaining_years']
    df['storey_x_lease_rem'] = df['mid_storey'] * df['lease_remaining_years']

    return df


train = feature_engineering(train)
test  = feature_engineering(test)

# ─── Categorical Encoding ─────────────────────────────────────────────────────
CAT_COLS = ['town', 'flat_type', 'flat_model', 'full_flat_type',
            'planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name']

# Label encode for XGBoost; LightGBM/CatBoost handle categoricals natively
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
    'storey_range', 'postal', 'bus_stop_name',
    # Raw lat/lon kept but encoded names dropped
] + CAT_COLS  # raw string versions replaced by _enc

FEATURES = [c for c in train.columns if c not in DROP_COLS]
print(f"\nNumber of features: {len(FEATURES)}")
print(FEATURES[:20], "...")

X      = train[FEATURES]
y      = train[TARGET]
X_test = test[FEATURES]

# ─── 80/20 Train/Validation Split ────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_tr):,}  |  Val size: {len(X_val):,}")

# ─── LightGBM ────────────────────────────────────────────────────────────────
print("\n=== LightGBM ===")
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
val_lgb  = lgb_model.predict(X_val)
lgb_rmse = root_mean_squared_error(y_val, val_lgb)
print(f"  LightGBM Val RMSE: {lgb_rmse:,.0f}")

# ─── XGBoost ─────────────────────────────────────────────────────────────────
print("\n=== XGBoost ===")
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
val_xgb  = xgb_model.predict(X_val)
xgb_rmse = root_mean_squared_error(y_val, val_xgb)
print(f"  XGBoost Val RMSE: {xgb_rmse:,.0f}")

# ─── CatBoost ────────────────────────────────────────────────────────────────
print("\n=== CatBoost ===")
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
val_cat  = cat_model.predict(X_val)
cat_rmse = root_mean_squared_error(y_val, val_cat)
print(f"  CatBoost Val RMSE: {cat_rmse:,.0f}")

# ─── Ensemble (Optimised Weighted Average on Val Set) ────────────────────────
from scipy.optimize import minimize

def ensemble_rmse(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = w[0]*val_lgb + w[1]*val_xgb + w[2]*val_cat
    return root_mean_squared_error(y_val, blend)

res = minimize(ensemble_rmse, [1, 1, 1], method='Nelder-Mead',
               options={'maxiter': 1000, 'xatol': 1e-6})
best_w     = res.x / res.x.sum()
val_blend  = best_w[0]*val_lgb + best_w[1]*val_xgb + best_w[2]*val_cat
blend_rmse = root_mean_squared_error(y_val, val_blend)

print(f"\n=== Ensemble ===")
print(f"  Best weights — LGB: {best_w[0]:.3f}, XGB: {best_w[1]:.3f}, CAT: {best_w[2]:.3f}")
print(f"  Ensemble Val RMSE: {blend_rmse:,.0f}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Validation RMSE Summary (20% hold-out) ===")
print(f"  LightGBM : {lgb_rmse:,.0f}")
print(f"  XGBoost  : {xgb_rmse:,.0f}")
print(f"  CatBoost : {cat_rmse:,.0f}")
print(f"  Ensemble : {blend_rmse:,.0f}")

# ─── Predict on test.csv ─────────────────────────────────────────────────────
preds_blend = (best_w[0] * lgb_model.predict(X_test) +
               best_w[1] * xgb_model.predict(X_test) +
               best_w[2] * cat_model.predict(X_test))

# ─── Submission ───────────────────────────────────────────────────────────────
sample = pd.read_csv('../data/sample_sub_reg.csv')
sub = pd.DataFrame({"Id": test[ID_COL], "Predicted": preds_blend})

sub.to_csv('../submission/submission_ensemble.csv', index=False)
print("\nSubmission saved to submission/submission_ensemble.csv")
print(sub.head())
