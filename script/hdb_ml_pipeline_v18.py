"""
HDB Resale Price Prediction Pipeline — v18
Models   : LightGBM · XGBoost · CatBoost (v15 Optuna params, 3-fold inner CV)
Ensemble : Nelder-Mead weight optimisation on OOF predictions
Output   : submission_v18_5fold.csv

Changes vs v15
──────────────
1. Added 1000 m spatial radius alongside existing 500 m and 2000 m.
   RADII_M = [500, 1000, 2000]

   Motivation: the gap between 500 m (block-level) and 2000 m (district-level)
   is coarse. 1000 m captures the immediate neighbourhood — typically 1–3
   adjacent HDB estates — which is a meaningful intermediate price signal.

   New feature added per fold:
     spatial_1000m_te  — smoothed mean price within 1000 m

   spatial_500m_psm stays at the finest radius (500 m) unchanged.

2. No postal encoding, no time-aware spatial (isolated experiment vs v15).

3. Total features per fold: 63 (was 62 in v15).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — v15 best hyperparameters (3-fold inner CV, 50 trials each)
# ══════════════════════════════════════════════════════════════════════════════

LGB_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'n_estimators':      5000,
    'bagging_freq':      1,
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      42,
    'learning_rate':     0.015823133257513698,
    'num_leaves':        179,
    'min_child_samples': 127,
    'feature_fraction':  0.45670099602309017,
    'bagging_fraction':  0.9490340827010041,
    'reg_alpha':         0.6354383365050222,
    'reg_lambda':        2.880007754943537e-05,
}

XGB_PARAMS = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'n_estimators':     5000,
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
    'learning_rate':    0.016417030028739503,
    'max_depth':        10,
    'min_child_weight': 24,
    'subsample':        0.7049619167454984,
    'colsample_bytree': 0.40012944717149146,
    'reg_alpha':        2.145570496484021e-05,
    'reg_lambda':       0.042439489227924364,
    'gamma':            2.027027628519129,
}

CAT_PARAMS = {
    'loss_function':       'RMSE',
    'iterations':          5000,
    'od_type':             'Iter',
    'od_wait':             100,
    'verbose':             0,
    'random_seed':         42,
    'task_type':           'CPU',
    'learning_rate':       0.03494303480441394,
    'depth':               10,
    'l2_leaf_reg':         0.032906695352702554,
    'random_strength':     1.149913856504512,
    'bagging_temperature': 0.935042332806602,
    'border_count':        136,
}

TARGET_ENC_COLS = ['town', 'planning_area']
LABEL_ENC_COLS  = ['flat_type', 'flat_model', 'mrt_name', 'pri_sch_name', 'sec_sch_name']
REDUNDANT_COLS  = ['floor_area_sqft', 'hdb_age', 'lower', 'upper', 'mid']
LOW_IMP_COLS    = [
    '1room_rental', '2room_rental', '3room_rental', 'other_room_rental',
    '1room_sold', 'studio_apartment_sold', 'multigen_sold',
    'residential', 'commercial', 'market_hawker',
    'multistorey_carpark', 'precinct_pavilion',
    'mrt_interchange', 'bus_interchange',
    'Tranc_Month',
    'affiliation', 'pri_sch_affiliation',
    'Latitude', 'Longitude',
]

LAT_M   = 111_000.0
LON_M   = 110_970.0
RADII_M = [500, 1000, 2000]

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
    print(f"  {label:<42}  RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  MAPE={mape:>6.2f}%  R²={r2:.4f}")


def optimise_weights(y_true, p1, p2, p3):
    def rmse_w(w):
        w = np.abs(w) / np.abs(w).sum()
        return root_mean_squared_error(y_true, w[0]*p1 + w[1]*p2 + w[2]*p3)
    res = minimize(rmse_w, [1, 1, 1], method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6})
    return np.abs(res.x) / np.abs(res.x).sum()


def _smooth(count, grp_mean, global_mean, k=10):
    s = 1.0 / (1.0 + np.exp(-(count - k) / k))
    return global_mean * (1 - s) + grp_mean * s


def _latlon_to_m(df):
    return np.column_stack([
        df['Latitude'].to_numpy() * LAT_M,
        df['Longitude'].to_numpy() * LON_M,
    ])


def target_encode_fold(df_tr, df_val, df_te, cols, target_col, k=10):
    """Leakage-free smoothed mean target encoding — stats from df_tr only."""
    global_mean = df_tr[target_col].mean()
    tr_enc, val_enc, te_enc = {}, {}, {}
    for col in cols:
        agg    = df_tr.groupby(col)[target_col].agg(['mean', 'count'])
        smooth = 1.0 / (1.0 + np.exp(-(agg['count'] - k) / k))
        enc    = global_mean * (1 - smooth) + agg['mean'] * smooth
        tr_enc[col  + '_te'] = df_tr[col].map(enc).fillna(global_mean).to_numpy()
        val_enc[col + '_te'] = df_val[col].map(enc).fillna(global_mean).to_numpy()
        te_enc[col  + '_te'] = df_te[col].map(enc).fillna(global_mean).to_numpy()
    return tr_enc, val_enc, te_enc


def spatial_radius_encode_fold(df_tr, df_val, df_te, target_col, radii_m=None, k=10):
    """Static spatial encoding — KD-tree built from df_tr only."""
    if radii_m is None:
        radii_m = RADII_M
    global_mean = df_tr[target_col].mean()
    prices_tr   = df_tr[target_col].to_numpy()
    psm_tr      = (df_tr[target_col] / df_tr['floor_area_sqm']).to_numpy()
    global_psm  = psm_tr.mean()
    coords_tr   = _latlon_to_m(df_tr)
    coords_val  = _latlon_to_m(df_val)
    coords_te   = _latlon_to_m(df_te)
    tree        = cKDTree(coords_tr)

    def _encode(query_coords, values_tr, global_val, radius):
        nbrs = tree.query_ball_point(query_coords, r=radius, workers=-1)
        out  = np.empty(len(query_coords))
        for i, n in enumerate(nbrs):
            out[i] = global_val if len(n) == 0 else _smooth(len(n), values_tr[n].mean(), global_val, k)
        return out

    tr_enc, val_enc, te_enc = {}, {}, {}
    for r in radii_m:
        feat = f'spatial_{r}m_te'
        tr_enc[feat]  = _encode(coords_tr,  prices_tr, global_mean, r)
        val_enc[feat] = _encode(coords_val, prices_tr, global_mean, r)
        te_enc[feat]  = _encode(coords_te,  prices_tr, global_mean, r)
    r0 = min(radii_m)
    feat = f'spatial_{r0}m_psm'
    tr_enc[feat]  = _encode(coords_tr,  psm_tr, global_psm, r0)
    val_enc[feat] = _encode(coords_val, psm_tr, global_psm, r0)
    te_enc[feat]  = _encode(coords_te,  psm_tr, global_psm, r0)
    return tr_enc, val_enc, te_enc


def attach_encodings(X_base, enc_dict):
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
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def feature_engineering(df, mall_dist_median):
    df = df.copy()
    df['lease_remaining_years'] = 99 - (df['Tranc_Year'] - df['lease_commence_date'])
    df['lease_remaining_pct']   = df['lease_remaining_years'] / 99.0
    df['tranc_period']          = df['Tranc_Year'] * 12 + df['Tranc_Month']
    df['storey_ratio']          = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan)
    df['is_high_floor']         = (df['mid_storey'] >= 20).astype(int)
    df['floor_band']            = pd.cut(df['mid_storey'],
                                         bins=[0, 5, 10, 15, 20, 30, 50, 999],
                                         labels=[1, 2, 3, 4, 5, 6, 7]).astype(float)
    df['log_mrt_dist']          = np.log1p(df['mrt_nearest_distance'])
    df['log_mall_dist']         = np.log1p(df['Mall_Nearest_Distance'].fillna(mall_dist_median))
    df['log_hawker_dist']       = np.log1p(df['Hawker_Nearest_Distance'])
    df['accessibility_score']   = (
        df['log_mrt_dist'] * 0.4 + df['log_mall_dist'] * 0.2 + df['log_hawker_dist'] * 0.2
    )
    for col in ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']:
        df[col] = df[col].fillna(0)
    df['school_quality']              = df['cutoff_point'].fillna(0) + df['affiliation'].fillna(0) * 10
    df['pri_school_quality']          = (df['pri_sch_affiliation'].fillna(0) * 10
                                         + 1 / (df['pri_sch_nearest_distance'] + 1))
    df['area_x_storey']               = df['floor_area_sqm'] * df['mid_storey']
    df['area_x_lease_rem']            = df['floor_area_sqm'] * df['lease_remaining_years']
    df['storey_x_lease_rem']          = df['mid_storey']     * df['lease_remaining_years']
    df['year_completed_x_floor_area'] = df['year_completed'] * df['floor_area_sqm']
    return df


mall_dist_median = train_raw['Mall_Nearest_Distance'].median()
train_fe = feature_engineering(train_raw, mall_dist_median)
test_fe  = feature_engineering(test_raw,  mall_dist_median)

# ══════════════════════════════════════════════════════════════════════════════
# LABEL ENCODING
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
# ══════════════════════════════════════════════════════════════════════════════

DROP_COLS = [
    TARGET, ID_COL,
    'Tranc_YearMonth', 'block', 'street_name', 'address', 'storey_range',
    'postal', 'bus_stop_name', 'full_flat_type',
    *REDUNDANT_COLS,
    *LOW_IMP_COLS,
    *TARGET_ENC_COLS,   # raw strings dropped; _te columns added per fold
    *LABEL_ENC_COLS,    # raw strings replaced by _enc columns
]

BASE_FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
SPATIAL_FEATS = [f'spatial_{r}m_te' for r in RADII_M] + \
                [f'spatial_{min(RADII_M)}m_psm']
n_total = len(BASE_FEATURES) + len(TARGET_ENC_COLS) + len(SPATIAL_FEATS)

print(f"\nBase features                : {len(BASE_FEATURES)}")
print(f"Target-encoded (per fold)   : {TARGET_ENC_COLS}")
print(f"Spatial-encoded (per fold)  : {SPATIAL_FEATS}")
print(f"Total features per fold     : {n_total}")

train_fe = train_fe.reset_index(drop=True)
test_fe  = test_fe.reset_index(drop=True)
y        = train_fe[TARGET]
X_test_base = test_fe[BASE_FEATURES]

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING — 5-fold OOF
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Training — {N_FOLDS}-fold OOF  (leak-free encoding per fold)")
print("═"*65)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_fe))
oof_xgb = np.zeros(len(train_fe))
oof_cat = np.zeros(len(train_fe))

pred_lgb = np.zeros(len(test_fe))
pred_xgb = np.zeros(len(test_fe))
pred_cat = np.zeros(len(test_fe))

imp_lgb_folds = []
imp_xgb_folds = []
imp_cat_folds = []
feature_names = None

for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(len(train_fe))), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}  train={len(tr_idx):,}  val={len(val_idx):,}")

    df_tr_f  = train_fe.iloc[tr_idx]
    df_val_f = train_fe.iloc[val_idx]

    # Target encoding: town, planning_area, postal — all from training fold only
    tr_enc, val_enc, te_enc = target_encode_fold(
        df_tr=df_tr_f, df_val=df_val_f, df_te=test_fe,
        cols=TARGET_ENC_COLS, target_col=TARGET,
    )

    # Spatial encoding
    print(f"    Computing spatial encodings (KD-tree)...", end=' ', flush=True)
    sp_tr, sp_val, sp_te = spatial_radius_encode_fold(
        df_tr=df_tr_f, df_val=df_val_f, df_te=test_fe, target_col=TARGET,
    )
    print("done")

    tr_enc  = {**tr_enc,  **sp_tr}
    val_enc = {**val_enc, **sp_val}
    te_enc  = {**te_enc,  **sp_te}

    X_tr  = attach_encodings(df_tr_f[BASE_FEATURES],  tr_enc)
    X_val = attach_encodings(df_val_f[BASE_FEATURES], val_enc)
    X_te  = attach_encodings(X_test_base,             te_enc)

    if feature_names is None:
        feature_names = list(X_tr.columns)

    y_tr  = y.iloc[tr_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    m_lgb = fit_lgb(X_tr, y_tr, X_val, y_val)
    oof_lgb[val_idx] = m_lgb.predict(X_val)
    pred_lgb        += m_lgb.predict(X_te) / N_FOLDS
    imp_lgb_folds.append(m_lgb.booster_.feature_importance(importance_type='gain'))
    print_metrics(f"    LGB fold {fold}", y_val, oof_lgb[val_idx])

    m_xgb = fit_xgb(X_tr, y_tr, X_val, y_val)
    oof_xgb[val_idx] = m_xgb.predict(X_val)
    pred_xgb        += m_xgb.predict(X_te) / N_FOLDS
    xgb_scores = m_xgb.get_booster().get_score(importance_type='gain')
    imp_xgb_folds.append([xgb_scores.get(f, 0.0) for f in feature_names])
    print_metrics(f"    XGB fold {fold}", y_val, oof_xgb[val_idx])

    m_cat = fit_cat(X_tr, y_tr, X_val, y_val)
    oof_cat[val_idx] = m_cat.predict(X_val)
    pred_cat        += m_cat.predict(X_te) / N_FOLDS
    imp_cat_folds.append(m_cat.get_feature_importance())
    print_metrics(f"    CAT fold {fold}", y_val, oof_cat[val_idx])

w         = optimise_weights(y, oof_lgb, oof_xgb, oof_cat)
oof_blend = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

V15_OOF_RMSE = 21_334

print("\n" + "═"*65)
print("Final Summary")
print("═"*65)
print_metrics("LightGBM  (OOF)", y, oof_lgb)
print_metrics("XGBoost   (OOF)", y, oof_xgb)
print_metrics("CatBoost  (OOF)", y, oof_cat)
print_metrics(f"Ensemble  (L={w[0]:.2f} X={w[1]:.2f} C={w[2]:.2f})", y, oof_blend)

v18_rmse = root_mean_squared_error(y, oof_blend)
delta    = V15_OOF_RMSE - v18_rmse
print(f"\n  v15 ensemble OOF RMSE : {V15_OOF_RMSE:,}")
print(f"  v18 ensemble OOF RMSE : {v18_rmse:,.0f}  "
      f"({'↓ improved by ' + f'{delta:,.0f}' if delta > 0 else '↑ regressed by ' + f'{abs(delta):,.0f}'})")
print("═"*65)

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

pred_final = w[0]*pred_lgb + w[1]*pred_xgb + w[2]*pred_cat
pd.DataFrame({"Id": test_raw[ID_COL], "Predicted": pred_final}).to_csv(
    '../submission/submission_v18_5fold.csv', index=False
)
print("\nSubmission saved: submission_v18_5fold.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("Feature Importance — 5-fold average gain (normalised to %)")
print("═"*65)

imp_lgb_arr = np.array(imp_lgb_folds).mean(axis=0)
imp_xgb_arr = np.array(imp_xgb_folds).mean(axis=0)
imp_cat_arr = np.array(imp_cat_folds).mean(axis=0)

imp_lgb_pct = 100 * imp_lgb_arr / imp_lgb_arr.sum()
imp_xgb_pct = 100 * imp_xgb_arr / imp_xgb_arr.sum() if imp_xgb_arr.sum() > 0 else imp_xgb_arr
imp_cat_pct = 100 * imp_cat_arr / imp_cat_arr.sum()

df_imp = pd.DataFrame({
    'feature': feature_names,
    'lgb_%':   imp_lgb_pct,
    'xgb_%':   imp_xgb_pct,
    'cat_%':   imp_cat_pct,
})
df_imp['mean_%'] = df_imp[['lgb_%', 'xgb_%', 'cat_%']].mean(axis=1)
df_imp = df_imp.sort_values('mean_%', ascending=False).reset_index(drop=True)

print(f"\n  {'Feature':<40} {'LGB%':>6}  {'XGB%':>6}  {'CAT%':>6}  {'Mean%':>6}")
print("  " + "-"*66)
for _, row in df_imp.head(35).iterrows():
    print(f"  {row['feature']:<40} {row['lgb_%']:>6.2f}  {row['xgb_%']:>6.2f}"
          f"  {row['cat_%']:>6.2f}  {row['mean_%']:>6.2f}")

low = df_imp[df_imp['mean_%'] < 0.10].sort_values('mean_%')
if len(low):
    print(f"\nLow-importance features (mean gain < 0.10%) — {len(low)} features:")
    for _, row in low.iterrows():
        print(f"  {row['feature']:<40} {row['lgb_%']:>6.2f}  {row['xgb_%']:>6.2f}"
              f"  {row['cat_%']:>6.2f}  {row['mean_%']:>6.2f}")
else:
    print("\nNo features with mean gain < 0.10% — all features contribute.")

# Show where the new 1000m feature lands
if 'spatial_1000m_te' in df_imp['feature'].values:
    row  = df_imp[df_imp['feature'] == 'spatial_1000m_te'].iloc[0]
    rank = df_imp[df_imp['feature'] == 'spatial_1000m_te'].index[0] + 1
    print(f"\nspatial_1000m_te  →  rank {rank}  |  mean gain {row['mean_%']:.2f}%  "
          f"(LGB {row['lgb_%']:.2f}%  XGB {row['xgb_%']:.2f}%  CAT {row['cat_%']:.2f}%)")
