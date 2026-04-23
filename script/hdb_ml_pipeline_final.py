"""
HDB Resale Price Prediction — Final Pipeline
Models   : LightGBM · XGBoost · CatBoost (all Optuna-tuned)
Ensemble : Nelder-Mead weight optimisation on validation predictions
Outputs  : Two submissions
             submission_final_8020.csv   — 80/20 random split
             submission_final_5fold.csv  — 5-fold OOF (more reliable)
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
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — tuned hyperparameters (Optuna, v7–v9)
# ══════════════════════════════════════════════════════════════════════════════

LGB_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'n_estimators':      5000,
    'bagging_freq':      1,
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      42,
    'learning_rate':     0.010021172250241975,
    'num_leaves':        323,
    'min_child_samples': 43,
    'feature_fraction':  0.5670125570814617,
    'bagging_fraction':  0.8889351795009706,
    'reg_alpha':         0.00017863857478743836,
    'reg_lambda':        0.00038653010503709084,
}

XGB_PARAMS = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'n_estimators':     5000,
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
    'learning_rate':    0.010508405331770552,
    'max_depth':        10,
    'min_child_weight': 10,
    'subsample':        0.7072958782974688,
    'colsample_bytree': 0.5050778304025615,
    'reg_alpha':        0.00235824395321555,
    'reg_lambda':       0.10765161235657794,
    'gamma':            0.2758542525827162,
}

CAT_PARAMS = {
    'loss_function':       'RMSE',
    'iterations':          5000,
    'od_type':             'Iter',
    'od_wait':             100,
    'verbose':             0,
    'random_seed':         42,
    'task_type':           'CPU',
    'learning_rate':       0.0318637615066506,
    'depth':               10,
    'l2_leaf_reg':         1.2695026389840243,
    'random_strength':     1.540606655478695,
    'bagging_temperature': 1.2874233422351384,
    'border_count':        205,
}

EARLY_STOP = 100
N_FOLDS    = 5

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(label, y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:<38}  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:>6.2f}%  R²={r2:.4f}")


def optimise_weights(y_true, p1, p2, p3):
    """Nelder-Mead ensemble weight optimisation. Returns normalised weights."""
    def rmse_w(w):
        w = np.abs(w) / np.abs(w).sum()
        return root_mean_squared_error(y_true, w[0]*p1 + w[1]*p2 + w[2]*p3)
    res = minimize(rmse_w, [1, 1, 1], method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6})
    return np.abs(res.x) / np.abs(res.x).sum()


def fit_lgb(X_tr, y_tr, X_val, y_val):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                     lgb.log_evaluation(0)])
    return m


def fit_xgb(X_tr, y_tr, X_val, y_val):
    m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOP, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def fit_cat(X_tr, y_tr, X_val, y_val):
    m = cb.CatBoostRegressor(**CAT_PARAMS)
    m.fit(X_tr, y_tr,
          eval_set=(X_val, y_val),
          early_stopping_rounds=EARLY_STOP,
          use_best_model=True,
          verbose=False)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv',  low_memory=False)

print(f"Train : {train.shape}")
print(f"Test  : {test.shape}")

TARGET = 'resale_price'
ID_COL = 'id'

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def feature_engineering(df, mall_dist_median):
    df = df.copy()

    # Time
    df['lease_remaining_years'] = 99 - (df['Tranc_Year'] - df['lease_commence_date'])
    df['lease_remaining_pct']   = df['lease_remaining_years'] / 99.0
    df['month_sin']             = np.sin(2 * np.pi * df['Tranc_Month'] / 12)
    df['month_cos']             = np.cos(2 * np.pi * df['Tranc_Month'] / 12)
    df['tranc_period']          = df['Tranc_Year'] * 12 + df['Tranc_Month']

    # Storey
    df['storey_ratio']  = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan)
    df['is_high_floor'] = (df['mid_storey'] >= 20).astype(int)
    df['floor_band']    = pd.cut(df['mid_storey'],
                                 bins=[0, 5, 10, 15, 20, 30, 50, 999],
                                 labels=[1, 2, 3, 4, 5, 6, 7]).astype(float)

    # Distances (log-transform reduces right-skew)
    df['log_mrt_dist']    = np.log1p(df['mrt_nearest_distance'])
    df['log_mall_dist']   = np.log1p(df['Mall_Nearest_Distance'].fillna(mall_dist_median))
    df['log_hawker_dist'] = np.log1p(df['Hawker_Nearest_Distance'])
    df['log_bus_dist']    = np.log1p(df['bus_stop_nearest_distance'])
    df['log_pri_dist']    = np.log1p(df['pri_sch_nearest_distance'])
    df['log_sec_dist']    = np.log1p(df['sec_sch_nearest_dist'])
    df['accessibility_score'] = (
        df['log_mrt_dist'] * 0.4 + df['log_mall_dist']   * 0.2 +
        df['log_hawker_dist'] * 0.2 + df['log_bus_dist'] * 0.2
    )

    for col in ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']:
        df[col] = df[col].fillna(0)

    # Building
    df['is_mrt_interchange'] = df['mrt_interchange'].fillna(0).astype(int)
    df['is_bus_interchange'] = df['bus_interchange'].fillna(0).astype(int)
    df['total_sold_units']   = df[['1room_sold', '2room_sold', '3room_sold', '4room_sold',
                                    '5room_sold', 'exec_sold', 'multigen_sold',
                                    'studio_apartment_sold']].sum(axis=1)
    df['total_rental_units'] = df[['1room_rental', '2room_rental',
                                    '3room_rental', 'other_room_rental']].sum(axis=1)
    df['sold_ratio']         = df['total_sold_units'] / df['total_dwelling_units'].replace(0, np.nan)

    # School quality
    df['school_quality']     = df['cutoff_point'].fillna(0) + df['affiliation'].fillna(0) * 10
    df['pri_school_quality'] = (df['pri_sch_affiliation'].fillna(0) * 10
                                + 1 / (df['pri_sch_nearest_distance'] + 1))

    # Binary flags
    for col in ['residential', 'commercial', 'market_hawker',
                'multistorey_carpark', 'precinct_pavilion']:
        df[col] = (df[col] == 'Y').astype(int)

    # Interactions
    df['area_x_storey']               = df['floor_area_sqm'] * df['mid_storey']
    df['area_x_lease_rem']            = df['floor_area_sqm'] * df['lease_remaining_years']
    df['storey_x_lease_rem']          = df['mid_storey']     * df['lease_remaining_years']
    df['year_completed_x_floor_area'] = df['year_completed'] * df['floor_area_sqm']

    return df


mall_dist_median = train['Mall_Nearest_Distance'].median()
train = feature_engineering(train, mall_dist_median)
test  = feature_engineering(test,  mall_dist_median)

# ══════════════════════════════════════════════════════════════════════════════
# ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def smoothed_target_encode(df_tr, df_te, col, target, k=10):
    global_mean = df_tr[target].mean()
    agg    = df_tr.groupby(col)[target].agg(['mean', 'count'])
    smooth = 1.0 / (1.0 + np.exp(-(agg['count'] - k) / k))
    enc    = global_mean * (1 - smooth) + agg['mean'] * smooth
    return df_tr[col].map(enc).fillna(global_mean), df_te[col].map(enc).fillna(global_mean)


TARGET_ENC_COLS = ['town', 'planning_area']
LABEL_ENC_COLS  = ['flat_type', 'flat_model', 'mrt_name', 'pri_sch_name', 'sec_sch_name']

for col in TARGET_ENC_COLS:
    train[col + '_te'], test[col + '_te'] = smoothed_target_encode(train, test, col, TARGET)

for col in LABEL_ENC_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
    le.fit(combined)
    train[col + '_enc'] = le.transform(train[col].astype(str))
    test[col + '_enc']  = le.transform(test[col].astype(str))

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

DROP_COLS = [
    TARGET, ID_COL,
    'Tranc_YearMonth', 'block', 'street_name', 'address', 'storey_range',
    'postal', 'bus_stop_name', 'full_flat_type',
    # Redundant: exact duplicates or near-perfect collinear pairs
    'floor_area_sqft', 'hdb_age', 'lower', 'upper', 'mid',
] + TARGET_ENC_COLS + LABEL_ENC_COLS

FEATURES = [c for c in train.columns if c not in DROP_COLS]
print(f"\nFeatures : {len(FEATURES)}")

X      = train[FEATURES].reset_index(drop=True)
y      = train[TARGET].reset_index(drop=True)
X_test = test[FEATURES]

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING A — 80 / 20 random split
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("Training A — 80/20 random split")
print("═"*65)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train={len(X_tr):,}  Val={len(X_val):,}\n")

m_lgb_a = fit_lgb(X_tr, y_tr, X_val, y_val);  val_lgb_a  = m_lgb_a.predict(X_val);  pred_lgb_a  = m_lgb_a.predict(X_test)
m_xgb_a = fit_xgb(X_tr, y_tr, X_val, y_val);  val_xgb_a  = m_xgb_a.predict(X_val);  pred_xgb_a  = m_xgb_a.predict(X_test)
m_cat_a = fit_cat(X_tr, y_tr, X_val, y_val);  val_cat_a  = m_cat_a.predict(X_val);  pred_cat_a  = m_cat_a.predict(X_test)

w_a       = optimise_weights(y_val, val_lgb_a, val_xgb_a, val_cat_a)
val_ens_a = w_a[0]*val_lgb_a + w_a[1]*val_xgb_a + w_a[2]*val_cat_a
y_val_a   = y_val.copy()   # preserve before OOF loop overwrites y_val

print_metrics("LightGBM",          y_val_a, val_lgb_a)
print_metrics("XGBoost",           y_val_a, val_xgb_a)
print_metrics("CatBoost",          y_val_a, val_cat_a)
print_metrics(
    f"Ensemble (L={w_a[0]:.2f} X={w_a[1]:.2f} C={w_a[2]:.2f})",
    y_val_a, val_ens_a
)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING B — 5-fold OOF
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Training B — {N_FOLDS}-fold OOF")
print("═"*65)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb  = np.zeros(len(X));  pred_lgb_b  = np.zeros(len(X_test))
oof_xgb  = np.zeros(len(X));  pred_xgb_b  = np.zeros(len(X_test))
oof_cat  = np.zeros(len(X));  pred_cat_b  = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}  train={len(tr_idx):,}  val={len(val_idx):,}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    m_lgb = fit_lgb(X_tr, y_tr, X_val, y_val)
    oof_lgb[val_idx] = m_lgb.predict(X_val);  pred_lgb_b += m_lgb.predict(X_test) / N_FOLDS
    print_metrics(f"    LGB fold {fold}", y_val, oof_lgb[val_idx])

    m_xgb = fit_xgb(X_tr, y_tr, X_val, y_val)
    oof_xgb[val_idx] = m_xgb.predict(X_val);  pred_xgb_b += m_xgb.predict(X_test) / N_FOLDS
    print_metrics(f"    XGB fold {fold}", y_val, oof_xgb[val_idx])

    m_cat = fit_cat(X_tr, y_tr, X_val, y_val)
    oof_cat[val_idx] = m_cat.predict(X_val);  pred_cat_b += m_cat.predict(X_test) / N_FOLDS
    print_metrics(f"    CAT fold {fold}", y_val, oof_cat[val_idx])

w_b       = optimise_weights(y, oof_lgb, oof_xgb, oof_cat)
oof_blend = w_b[0]*oof_lgb + w_b[1]*oof_xgb + w_b[2]*oof_cat

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("Final Summary")
print("═"*65)
print("\n  ── Training A (80/20 split) ──")
print_metrics("LightGBM",          y_val_a, val_lgb_a)
print_metrics("XGBoost",           y_val_a, val_xgb_a)
print_metrics("CatBoost",          y_val_a, val_cat_a)
print_metrics(
    f"Ensemble (L={w_a[0]:.2f} X={w_a[1]:.2f} C={w_a[2]:.2f})",
    y_val_a, val_ens_a
)

print(f"\n  ── Training B ({N_FOLDS}-fold OOF) ──")
print_metrics("LightGBM  (OOF)",       y, oof_lgb)
print_metrics("XGBoost   (OOF)",       y, oof_xgb)
print_metrics("CatBoost  (OOF)",       y, oof_cat)
print_metrics(
    f"Ensemble  (L={w_b[0]:.2f} X={w_b[1]:.2f} C={w_b[2]:.2f})",
    y, oof_blend
)
print("═"*65)

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════════════

pred_a = w_a[0]*pred_lgb_a + w_a[1]*pred_xgb_a + w_a[2]*pred_cat_a
pred_b = w_b[0]*pred_lgb_b + w_b[1]*pred_xgb_b + w_b[2]*pred_cat_b

sub_a = pd.DataFrame({"Id": test[ID_COL], "Predicted": pred_a})
sub_b = pd.DataFrame({"Id": test[ID_COL], "Predicted": pred_b})

sub_a.to_csv('../submission/submission_final_8020.csv',  index=False)
sub_b.to_csv('../submission/submission_final_5fold.csv', index=False)

print("\nSubmissions saved:")
print("  submission_final_8020.csv   (80/20 split)")
print("  submission_final_5fold.csv  (5-fold OOF  ← recommended)")
