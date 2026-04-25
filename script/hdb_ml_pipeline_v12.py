"""
HDB Resale Price Prediction Pipeline — v12
Models   : LightGBM · XGBoost · CatBoost (all Optuna-tuned)
Ensemble : Nelder-Mead weight optimisation on validation predictions
Outputs  : Two submissions
             submission_v12_8020.csv  — 80/20 random split
             submission_v12_5fold.csv — 5-fold OOF (recommended)

Key change vs v11 — four spatial geohash features (all leakage-free):
  geohash6_te     : smoothed mean resale_price per geohash6 cell  (~1.2 km × 0.6 km)
  geohash7_te     : smoothed mean resale_price per geohash7 cell  (~150 m × 150 m)
  geohash7_psm_te : smoothed mean price-per-sqm per geohash7 cell
  geo6_nbr_price  : smoothed mean price of a cell + its 8 neighbours
                    (eliminates sharp boundary discontinuities between cells)

All stats are computed from training rows only; val/test rows are mapped
using those stats, so there is zero leakage — same guarantee as the
town/planning_area target encoding from v11.

Requires: pip install pygeohash
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
import pygeohash as gh
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
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

TARGET_ENC_COLS  = ['town', 'planning_area']   # encoded per-fold (leakage-free)
SPATIAL_ENC_COLS = ['geohash6', 'geohash7']    # raw hash strings; _te columns added per-fold
LABEL_ENC_COLS   = ['flat_type', 'flat_model', 'mrt_name', 'pri_sch_name', 'sec_sch_name']
REDUNDANT_COLS   = ['floor_area_sqft', 'hdb_age', 'lower', 'upper', 'mid']

EARLY_STOP = 100
N_FOLDS    = 5
TARGET     = 'resale_price'
ID_COL     = 'id'

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
    def rmse_w(w):
        w = np.abs(w) / np.abs(w).sum()
        return root_mean_squared_error(y_true, w[0]*p1 + w[1]*p2 + w[2]*p3)
    res = minimize(rmse_w, [1, 1, 1], method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6})
    return np.abs(res.x) / np.abs(res.x).sum()


def _smooth_enc(agg, global_mean, k=10):
    """
    Bayesian / sigmoid smoothing toward the global mean.
        s       = sigmoid(count - k)
        encoded = global_mean × (1 - s) + group_mean × s
    Groups with fewer than k transactions are pulled toward global_mean.
    """
    s = 1.0 / (1.0 + np.exp(-(agg['count'] - k) / k))
    return global_mean * (1 - s) + agg['mean'] * s


def target_encode_fold(df_tr, df_val, df_te, cols, target_col, k=10):
    """
    Leakage-free smoothed mean target encoding.

    Statistics (group mean, count) are derived exclusively from df_tr.
    df_val and df_te are encoded by mapping those statistics — they never
    contribute to their own encoding values.

    Parameters
    ----------
    df_tr  : training portion of train (with target column present)
    df_val : validation portion of train (target used only for metrics, not encoding)
    df_te  : test dataframe
    cols   : list of categorical column names to encode
    target_col : name of the target column in df_tr
    k      : smoothing factor (default 10)

    Returns
    -------
    tr_enc, val_enc, te_enc : dicts of {col_te: np.array}
    """
    global_mean = df_tr[target_col].mean()
    tr_enc, val_enc, te_enc = {}, {}, {}

    for col in cols:
        agg = df_tr.groupby(col)[target_col].agg(['mean', 'count'])
        enc = _smooth_enc(agg, global_mean, k)

        tr_enc[col  + '_te'] = df_tr[col].map(enc).fillna(global_mean).to_numpy()
        val_enc[col + '_te'] = df_val[col].map(enc).fillna(global_mean).to_numpy()
        te_enc[col  + '_te'] = df_te[col].map(enc).fillna(global_mean).to_numpy()

    return tr_enc, val_enc, te_enc


def geohash_encode_fold(df_tr, df_val, df_te, target_col, k=10):
    """
    Leakage-free geohash spatial encoding.  All four features are computed
    from df_tr only and then mapped to df_val / df_te.

    Features produced
    -----------------
    geohash6_te     : smoothed mean resale_price per geohash6 cell
    geohash7_te     : smoothed mean resale_price per geohash7 cell
    geohash7_psm_te : smoothed mean price-per-sqm per geohash7 cell
    geo6_nbr_price  : smoothed mean price of geohash6 cell + its 8 neighbours
                      (pygeohash.neighbors returns the 8 adjacent cells)

    Parameters
    ----------
    df_tr, df_val, df_te : DataFrames that already contain 'geohash6', 'geohash7'
                           columns (added in feature_engineering).
    target_col           : name of the target column (present only in df_tr).
    k                    : smoothing factor (default 10).

    Returns
    -------
    tr_enc, val_enc, te_enc : dicts of {feature_name: np.array}
    """
    global_mean = df_tr[target_col].mean()

    # ── geohash6_te ──────────────────────────────────────────────────────────
    agg6  = df_tr.groupby('geohash6')[target_col].agg(['mean', 'count'])
    enc6  = _smooth_enc(agg6, global_mean, k)

    # ── geohash7_te ──────────────────────────────────────────────────────────
    agg7  = df_tr.groupby('geohash7')[target_col].agg(['mean', 'count'])
    enc7  = _smooth_enc(agg7, global_mean, k)

    # ── geohash7_psm_te ──────────────────────────────────────────────────────
    psm        = df_tr[target_col] / df_tr['floor_area_sqm']
    global_psm = psm.mean()
    agg7p      = psm.groupby(df_tr['geohash7']).agg(['mean', 'count'])
    enc7p      = _smooth_enc(agg7p, global_psm, k)

    # ── geo6_nbr_price ───────────────────────────────────────────────────────
    # For each unique geohash6 code, collect the prices from that cell AND its
    # 8 neighbours, then compute the smoothed mean of that expanded pool.
    # This removes the sharp price discontinuity at cell boundaries.
    unique_g6 = set(df_tr['geohash6'].unique()) | set(df_val['geohash6'].unique()) | \
                set(df_te['geohash6'].unique())

    # Build neighbour lookup: code → list of [code] + 8 neighbours
    nbr_dict = {}
    for code in unique_g6:
        try:
            nbrs = gh.neighbors(code)             # returns dict of 8 directions
            nbr_dict[code] = [code] + list(nbrs.values())
        except Exception:
            nbr_dict[code] = [code]

    # Compute neighbourhood mean from df_tr only
    g6_series   = df_tr['geohash6'].to_numpy()
    price_series = df_tr[target_col].to_numpy()

    # cell-level stats (reuse agg6 counts for smoothing denominator)
    nbr_enc = {}
    for code in unique_g6:
        neighbours = nbr_dict.get(code, [code])
        mask       = np.isin(g6_series, neighbours)
        if mask.sum() == 0:
            nbr_enc[code] = global_mean
        else:
            n     = mask.sum()
            grp_m = price_series[mask].mean()
            s     = 1.0 / (1.0 + np.exp(-(n - k) / k))
            nbr_enc[code] = global_mean * (1 - s) + grp_m * s

    def _map(col_series):
        return col_series.map(nbr_enc).fillna(global_mean).to_numpy()

    tr_enc = {
        'geohash6_te':     df_tr['geohash6'].map(enc6).fillna(global_mean).to_numpy(),
        'geohash7_te':     df_tr['geohash7'].map(enc7).fillna(global_mean).to_numpy(),
        'geohash7_psm_te': df_tr['geohash7'].map(enc7p).fillna(global_psm).to_numpy(),
        'geo6_nbr_price':  _map(df_tr['geohash6']),
    }
    val_enc = {
        'geohash6_te':     df_val['geohash6'].map(enc6).fillna(global_mean).to_numpy(),
        'geohash7_te':     df_val['geohash7'].map(enc7).fillna(global_mean).to_numpy(),
        'geohash7_psm_te': df_val['geohash7'].map(enc7p).fillna(global_psm).to_numpy(),
        'geo6_nbr_price':  _map(df_val['geohash6']),
    }
    te_enc = {
        'geohash6_te':     df_te['geohash6'].map(enc6).fillna(global_mean).to_numpy(),
        'geohash7_te':     df_te['geohash7'].map(enc7).fillna(global_mean).to_numpy(),
        'geohash7_psm_te': df_te['geohash7'].map(enc7p).fillna(global_psm).to_numpy(),
        'geo6_nbr_price':  _map(df_te['geohash6']),
    }

    return tr_enc, val_enc, te_enc


def attach_encodings(X_base, enc_dict):
    """Attach a dict of {col: np.array} as new columns to a copy of X_base."""
    X = X_base.copy().reset_index(drop=True)
    for col, vals in enc_dict.items():
        X[col] = vals
    return X


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

train_raw = pd.read_csv('../data/train.csv', low_memory=False)
test_raw  = pd.read_csv('../data/test.csv',  low_memory=False)

print(f"Train : {train_raw.shape}")
print(f"Test  : {test_raw.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (no target encoding here)
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

    # Distances
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

    # Geohash  (raw string codes; target-encoded per fold in leakage-free manner)
    df['geohash6'] = df.apply(
        lambda r: gh.encode(r['Latitude'], r['Longitude'], precision=6), axis=1
    )
    df['geohash7'] = df.apply(
        lambda r: gh.encode(r['Latitude'], r['Longitude'], precision=7), axis=1
    )

    return df


mall_dist_median = train_raw['Mall_Nearest_Distance'].median()
train_fe = feature_engineering(train_raw, mall_dist_median)
test_fe  = feature_engineering(test_raw,  mall_dist_median)

# ══════════════════════════════════════════════════════════════════════════════
# LABEL ENCODING  (global — safe, target not involved)
# ══════════════════════════════════════════════════════════════════════════════

for col in LABEL_ENC_COLS:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)],
                         ignore_index=True)
    le.fit(combined)
    train_fe[col + '_enc'] = le.transform(train_fe[col].astype(str))
    test_fe[col  + '_enc'] = le.transform(test_fe[col].astype(str))

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# town / planning_area / geohash6 / geohash7 excluded from BASE_FEATURES —
# they are added as _te columns inside each training phase (leakage-free).
# ══════════════════════════════════════════════════════════════════════════════

DROP_COLS = [
    TARGET, ID_COL,
    'Tranc_YearMonth', 'block', 'street_name', 'address', 'storey_range',
    'postal', 'bus_stop_name', 'full_flat_type',
    *REDUNDANT_COLS,
    *TARGET_ENC_COLS,    # raw strings dropped; _te columns added per-fold
    *SPATIAL_ENC_COLS,   # raw geohash strings dropped; _te columns added per-fold
    *LABEL_ENC_COLS,     # raw strings replaced by _enc columns
]

BASE_FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
print(f"\nBase features (before encoding)         : {len(BASE_FEATURES)}")
print(f"Target-encoded columns added per fold   : {TARGET_ENC_COLS}")
print(f"Geohash-encoded columns added per fold  : geohash6_te, geohash7_te, "
      f"geohash7_psm_te, geo6_nbr_price")
print(f"Total features per fold                 : {len(BASE_FEATURES) + len(TARGET_ENC_COLS) + 4}")

# Aligned target and reset index for iloc-based indexing
train_fe = train_fe.reset_index(drop=True)
test_fe  = test_fe.reset_index(drop=True)
y        = train_fe[TARGET]

X_test_base = test_fe[BASE_FEATURES]   # base test features (no encoding yet)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING A — 80/20 random split  (leakage-free encoding)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("Training A — 80/20 random split")
print("═"*65)

tr_idx_a, val_idx_a = train_test_split(
    np.arange(len(train_fe)), test_size=0.20, random_state=42
)

df_tr_a  = train_fe.iloc[tr_idx_a]
df_val_a = train_fe.iloc[val_idx_a]

# Target encoding (town, planning_area)
tr_enc_a, val_enc_a, te_enc_a = target_encode_fold(
    df_tr=df_tr_a, df_val=df_val_a, df_te=test_fe,
    cols=TARGET_ENC_COLS, target_col=TARGET,
)

# Geohash encoding (4 spatial features)
gh_tr_a, gh_val_a, gh_te_a = geohash_encode_fold(
    df_tr=df_tr_a, df_val=df_val_a, df_te=test_fe, target_col=TARGET,
)

# Merge encoding dicts
tr_enc_a  = {**tr_enc_a,  **gh_tr_a}
val_enc_a = {**val_enc_a, **gh_val_a}
te_enc_a  = {**te_enc_a,  **gh_te_a}

X_tr_a   = attach_encodings(df_tr_a[BASE_FEATURES],  tr_enc_a)
X_val_a  = attach_encodings(df_val_a[BASE_FEATURES], val_enc_a)
X_test_a = attach_encodings(X_test_base, te_enc_a)

y_tr_a   = y.iloc[tr_idx_a].reset_index(drop=True)
y_val_a  = y.iloc[val_idx_a].reset_index(drop=True)

print(f"Train={len(X_tr_a):,}  Val={len(X_val_a):,}  Features={X_tr_a.shape[1]}\n")

m_lgb_a = fit_lgb(X_tr_a, y_tr_a, X_val_a, y_val_a)
m_xgb_a = fit_xgb(X_tr_a, y_tr_a, X_val_a, y_val_a)
m_cat_a = fit_cat(X_tr_a, y_tr_a, X_val_a, y_val_a)

val_lgb_a  = m_lgb_a.predict(X_val_a)
val_xgb_a  = m_xgb_a.predict(X_val_a)
val_cat_a  = m_cat_a.predict(X_val_a)
pred_lgb_a = m_lgb_a.predict(X_test_a)
pred_xgb_a = m_xgb_a.predict(X_test_a)
pred_cat_a = m_cat_a.predict(X_test_a)

w_a       = optimise_weights(y_val_a, val_lgb_a, val_xgb_a, val_cat_a)
val_ens_a = w_a[0]*val_lgb_a + w_a[1]*val_xgb_a + w_a[2]*val_cat_a

print_metrics("LightGBM",  y_val_a, val_lgb_a)
print_metrics("XGBoost",   y_val_a, val_xgb_a)
print_metrics("CatBoost",  y_val_a, val_cat_a)
print_metrics(f"Ensemble (L={w_a[0]:.2f} X={w_a[1]:.2f} C={w_a[2]:.2f})",
              y_val_a, val_ens_a)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING B — 5-fold OOF  (leakage-free encoding per fold)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Training B — {N_FOLDS}-fold OOF")
print("═"*65)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_fe))
oof_xgb = np.zeros(len(train_fe))
oof_cat = np.zeros(len(train_fe))

pred_lgb_b = np.zeros(len(test_fe))
pred_xgb_b = np.zeros(len(test_fe))
pred_cat_b = np.zeros(len(test_fe))

for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(len(train_fe))), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}  train={len(tr_idx):,}  val={len(val_idx):,}")

    df_tr_f  = train_fe.iloc[tr_idx]
    df_val_f = train_fe.iloc[val_idx]

    # Each fold computes its own independent target encoding
    # → validation rows never influence their own encoding
    tr_enc, val_enc, te_enc = target_encode_fold(
        df_tr=df_tr_f, df_val=df_val_f, df_te=test_fe,
        cols=TARGET_ENC_COLS, target_col=TARGET,
    )

    # Each fold computes its own independent geohash encoding
    gh_tr, gh_val, gh_te = geohash_encode_fold(
        df_tr=df_tr_f, df_val=df_val_f, df_te=test_fe, target_col=TARGET,
    )

    # Merge encoding dicts
    tr_enc  = {**tr_enc,  **gh_tr}
    val_enc = {**val_enc, **gh_val}
    te_enc  = {**te_enc,  **gh_te}

    X_tr  = attach_encodings(df_tr_f[BASE_FEATURES],  tr_enc)
    X_val = attach_encodings(df_val_f[BASE_FEATURES], val_enc)
    X_te  = attach_encodings(X_test_base,             te_enc)

    y_tr  = y.iloc[tr_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    m_lgb = fit_lgb(X_tr, y_tr, X_val, y_val)
    oof_lgb[val_idx] = m_lgb.predict(X_val)
    pred_lgb_b      += m_lgb.predict(X_te) / N_FOLDS
    print_metrics(f"    LGB fold {fold}", y_val, oof_lgb[val_idx])

    m_xgb = fit_xgb(X_tr, y_tr, X_val, y_val)
    oof_xgb[val_idx] = m_xgb.predict(X_val)
    pred_xgb_b      += m_xgb.predict(X_te) / N_FOLDS
    print_metrics(f"    XGB fold {fold}", y_val, oof_xgb[val_idx])

    m_cat = fit_cat(X_tr, y_tr, X_val, y_val)
    oof_cat[val_idx] = m_cat.predict(X_val)
    pred_cat_b      += m_cat.predict(X_te) / N_FOLDS
    print_metrics(f"    CAT fold {fold}", y_val, oof_cat[val_idx])

w_b       = optimise_weights(y, oof_lgb, oof_xgb, oof_cat)
oof_blend = w_b[0]*oof_lgb + w_b[1]*oof_xgb + w_b[2]*oof_cat

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

rmse_a = root_mean_squared_error(y_val_a, val_ens_a)
rmse_b = root_mean_squared_error(y,       oof_blend)

print("\n" + "═"*65)
print("Final Summary")
print("═"*65)

print("\n  ── Training A (80/20, leakage-free) ──")
print_metrics("LightGBM",  y_val_a, val_lgb_a)
print_metrics("XGBoost",   y_val_a, val_xgb_a)
print_metrics("CatBoost",  y_val_a, val_cat_a)
print_metrics(f"Ensemble (L={w_a[0]:.2f} X={w_a[1]:.2f} C={w_a[2]:.2f})",
              y_val_a, val_ens_a)

print(f"\n  ── Training B ({N_FOLDS}-fold OOF, leakage-free) ──")
print_metrics("LightGBM  (OOF)", y, oof_lgb)
print_metrics("XGBoost   (OOF)", y, oof_xgb)
print_metrics("CatBoost  (OOF)", y, oof_cat)
print_metrics(f"Ensemble  (L={w_b[0]:.2f} X={w_b[1]:.2f} C={w_b[2]:.2f})",
              y, oof_blend)

print(f"\n  Ensemble RMSE — 80/20 : {rmse_a:,.0f}")
print(f"  Ensemble RMSE — 5-fold: {rmse_b:,.0f}")
print("═"*65)

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════════════

pred_a = w_a[0]*pred_lgb_a + w_a[1]*pred_xgb_a + w_a[2]*pred_cat_a
pred_b = w_b[0]*pred_lgb_b + w_b[1]*pred_xgb_b + w_b[2]*pred_cat_b

pd.DataFrame({"Id": test_raw[ID_COL], "Predicted": pred_a}).to_csv(
    '../submission/submission_v12_8020.csv', index=False
)
pd.DataFrame({"Id": test_raw[ID_COL], "Predicted": pred_b}).to_csv(
    '../submission/submission_v12_5fold.csv', index=False
)

print("\nSubmissions saved:")
print("  submission_v12_8020.csv   (80/20 split)")
print("  submission_v12_5fold.csv  (5-fold OOF  ← recommended)")
