"""
HDB Resale Price Prediction Pipeline — v15
Models   : LightGBM · XGBoost · CatBoost (all re-tuned with 3-fold inner CV)
Ensemble : Nelder-Mead weight optimisation on OOF predictions
Output   : submission_v15_5fold.csv

Changes vs v14
──────────────
1. HPO objective upgraded from single 80/20 split → 3-fold inner CV.
   Each Optuna trial averages RMSE across 3 held-out folds instead of one,
   reducing trial variance and finding more robust hyperparameters.

2. All three models re-tuned. CatBoost reused v7/v8 params through v14;
   this is the first re-tune since spatial + leak-free target encoding
   were introduced in v13/v14.

3. Spatial features are pre-computed once on full training data for the HPO
   proxy (minor self-inclusion bias, acceptable since HPO only needs relative
   ranking of param configs, not absolute RMSE). The final OOF reuses the
   per-fold leak-free spatial + target encoding from v14.
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
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_HPO_TRIALS  = 50    # Optuna trials per model
N_HPO_FOLDS   = 3     # inner CV folds for HPO objective
N_OOF_FOLDS   = 5     # outer OOF folds for final evaluation
EARLY_STOP    = 100
N_ESTIMATORS  = 3000  # cap for HPO trials (early stopping determines actual count)
N_EST_FINAL   = 5000  # cap for final OOF (more budget, same early stopping)

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
RADII_M = [500, 2000]

TARGET = 'resale_price'
ID_COL = 'id'

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
    """Leakage-free continuous spatial encoding — KD-tree built from df_tr only."""
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
        neighbours = tree.query_ball_point(query_coords, r=radius, workers=-1)
        out = np.empty(len(query_coords))
        for i, nbr in enumerate(neighbours):
            out[i] = global_val if len(nbr) == 0 else _smooth(len(nbr), values_tr[nbr].mean(), global_val, k)
        return out

    tr_enc, val_enc, te_enc = {}, {}, {}
    for r in radii_m:
        feat = f'spatial_{r}m_te'
        tr_enc[feat]  = _encode(coords_tr,  prices_tr, global_mean, r)
        val_enc[feat] = _encode(coords_val, prices_tr, global_mean, r)
        te_enc[feat]  = _encode(coords_te,  prices_tr, global_mean, r)
    r0   = min(radii_m)
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


def fit_lgb(X_tr, y_tr, X_val, y_val, params, n_est):
    p = {**params, 'n_estimators': n_est}
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                     lgb.log_evaluation(0)])
    return m


def fit_xgb(X_tr, y_tr, X_val, y_val, params, n_est):
    p = {**params, 'n_estimators': n_est}
    m = xgb.XGBRegressor(**p, early_stopping_rounds=EARLY_STOP, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def fit_cat(X_tr, y_tr, X_val, y_val, params, n_est):
    p = {**params, 'iterations': n_est}
    m = cb.CatBoostRegressor(**p)
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
# LABEL ENCODING  (global — target not involved, safe)
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
    *TARGET_ENC_COLS,
    *LABEL_ENC_COLS,
]

BASE_FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
SPATIAL_FEATS = [f'spatial_{r}m_te' for r in RADII_M] + [f'spatial_{min(RADII_M)}m_psm']

print(f"\nBase features                : {len(BASE_FEATURES)}")
print(f"Target-encoded (per fold)   : {TARGET_ENC_COLS}")
print(f"Spatial-encoded (per fold)  : {SPATIAL_FEATS}")
print(f"Total features per fold     : {len(BASE_FEATURES) + len(TARGET_ENC_COLS) + len(SPATIAL_FEATS)}")

train_fe = train_fe.reset_index(drop=True)
test_fe  = test_fe.reset_index(drop=True)
y        = train_fe[TARGET]

# ══════════════════════════════════════════════════════════════════════════════
# HPO PROXY DATASET
# Pre-compute spatial + target encoding once on the full training set.
# Used for HPO inner folds only — avoids running KD-tree inside every trial.
# Minor self-inclusion bias is acceptable; HPO only needs relative param ranking.
# ══════════════════════════════════════════════════════════════════════════════

print("\nPre-computing HPO proxy encodings (full training data)...")

# Global target encoding (slight leak — validation rows' prices contribute to their own encoding)
global_mean = y.mean()
hpo_te_cols = []
for col in TARGET_ENC_COLS:
    agg    = train_fe.groupby(col)[TARGET].agg(['mean', 'count'])
    smooth = 1.0 / (1.0 + np.exp(-(agg['count'] - 10) / 10))
    enc    = global_mean * (1 - smooth) + agg['mean'] * smooth
    train_fe[col + '_te_proxy'] = train_fe[col].map(enc).fillna(global_mean)
    test_fe[col  + '_te_proxy'] = test_fe[col].map(enc).fillna(global_mean)
    hpo_te_cols.append(col + '_te_proxy')

# Global spatial encoding (slight self-inclusion bias)
prices_all = train_fe[TARGET].to_numpy()
psm_all    = (train_fe[TARGET] / train_fe['floor_area_sqm']).to_numpy()
coords_all = _latlon_to_m(train_fe)
coords_te  = _latlon_to_m(test_fe)
tree_all   = cKDTree(coords_all)

print("  Building global KD-tree...", end=' ', flush=True)
for r in RADII_M:
    feat = f'spatial_{r}m_te_proxy'
    nbrs = tree_all.query_ball_point(coords_all, r=r, workers=-1)
    vals = np.array([
        global_mean if len(n) == 0 else _smooth(len(n), prices_all[n].mean(), global_mean)
        for n in nbrs
    ])
    train_fe[feat] = vals
    nbrs_te = tree_all.query_ball_point(coords_te, r=r, workers=-1)
    test_fe[feat]  = np.array([
        global_mean if len(n) == 0 else _smooth(len(n), prices_all[n].mean(), global_mean)
        for n in nbrs_te
    ])

r0   = min(RADII_M)
feat = f'spatial_{r0}m_psm_proxy'
nbrs = tree_all.query_ball_point(coords_all, r=r0, workers=-1)
train_fe[feat] = np.array([
    psm_all.mean() if len(n) == 0 else _smooth(len(n), psm_all[n].mean(), psm_all.mean())
    for n in nbrs
])
nbrs_te = tree_all.query_ball_point(coords_te, r=r0, workers=-1)
test_fe[feat]  = np.array([
    psm_all.mean() if len(n) == 0 else _smooth(len(n), psm_all[n].mean(), psm_all.mean())
    for n in nbrs_te
])
print("done")

# HPO feature set = base + proxy target encoding + proxy spatial
PROXY_SPATIAL_FEATS = [f'spatial_{r}m_te_proxy' for r in RADII_M] + [f'spatial_{r0}m_psm_proxy']
PROXY_TE_FEATS      = [col + '_te_proxy' for col in TARGET_ENC_COLS]
HPO_FEATURES        = BASE_FEATURES + PROXY_TE_FEATS + PROXY_SPATIAL_FEATS

X_hpo      = train_fe[HPO_FEATURES].reset_index(drop=True)
y_hpo      = y.reset_index(drop=True)
hpo_kf     = KFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=0)

print(f"\nHPO proxy dataset: {X_hpo.shape[1]} features, {N_HPO_FOLDS}-fold inner CV")

# ══════════════════════════════════════════════════════════════════════════════
# HPO OBJECTIVE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def cv_rmse_lgb(params):
    scores = []
    for tr_idx, val_idx in hpo_kf.split(X_hpo):
        m = lgb.LGBMRegressor(**params, n_estimators=N_ESTIMATORS)
        m.fit(X_hpo.iloc[tr_idx], y_hpo.iloc[tr_idx],
              eval_set=[(X_hpo.iloc[val_idx], y_hpo.iloc[val_idx])],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                         lgb.log_evaluation(0)])
        scores.append(root_mean_squared_error(y_hpo.iloc[val_idx], m.predict(X_hpo.iloc[val_idx])))
    return np.mean(scores)


def cv_rmse_xgb(params):
    scores = []
    for tr_idx, val_idx in hpo_kf.split(X_hpo):
        m = xgb.XGBRegressor(**params, n_estimators=N_ESTIMATORS,
                              early_stopping_rounds=EARLY_STOP, verbosity=0)
        m.fit(X_hpo.iloc[tr_idx], y_hpo.iloc[tr_idx],
              eval_set=[(X_hpo.iloc[val_idx], y_hpo.iloc[val_idx])],
              verbose=False)
        scores.append(root_mean_squared_error(y_hpo.iloc[val_idx], m.predict(X_hpo.iloc[val_idx])))
    return np.mean(scores)


def cv_rmse_cat(params):
    scores = []
    for tr_idx, val_idx in hpo_kf.split(X_hpo):
        m = cb.CatBoostRegressor(**params, iterations=N_ESTIMATORS)
        m.fit(X_hpo.iloc[tr_idx], y_hpo.iloc[tr_idx],
              eval_set=(X_hpo.iloc[val_idx], y_hpo.iloc[val_idx]),
              early_stopping_rounds=EARLY_STOP,
              use_best_model=True,
              verbose=False)
        scores.append(root_mean_squared_error(y_hpo.iloc[val_idx], m.predict(X_hpo.iloc[val_idx])))
    return np.mean(scores)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1a — LightGBM HPO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Phase 1a — LightGBM HPO  ({N_HPO_TRIALS} trials, {N_HPO_FOLDS}-fold inner CV)")
print("═"*65)

LGB_BASE = {
    'objective':   'regression',
    'metric':      'rmse',
    'verbose':     -1,
    'n_jobs':      -1,
    'random_state': 42,
    'bagging_freq': 1,
}

def lgb_objective(trial):
    params = {
        **LGB_BASE,
        'learning_rate':     trial.suggest_float('learning_rate',     0.005, 0.10,  log=True),
        'num_leaves':        trial.suggest_int  ('num_leaves',        50,    600),
        'min_child_samples': trial.suggest_int  ('min_child_samples', 5,     150),
        'feature_fraction':  trial.suggest_float('feature_fraction',  0.4,   1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.4,   1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',         1e-5,  5.0,   log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda',        1e-5,  5.0,   log=True),
    }
    return cv_rmse_lgb(params)

lgb_study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=N_HPO_TRIALS, show_progress_bar=True)

BEST_LGB_PARAMS = {**LGB_BASE, **lgb_study.best_params}
print(f"\nBest LightGBM  (trial #{lgb_study.best_trial.number}, "
      f"3-fold CV RMSE = {lgb_study.best_value:,.0f}):")
for k, v in lgb_study.best_params.items():
    print(f"  {k:<25} = {v}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1b — XGBoost HPO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Phase 1b — XGBoost HPO  ({N_HPO_TRIALS} trials, {N_HPO_FOLDS}-fold inner CV)")
print("═"*65)

XGB_BASE = {
    'objective':   'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs':      -1,
}

def xgb_objective(trial):
    params = {
        **XGB_BASE,
        'learning_rate':    trial.suggest_float('learning_rate',    0.005, 0.10,  log=True),
        'max_depth':        trial.suggest_int  ('max_depth',        4,     12),
        'min_child_weight': trial.suggest_int  ('min_child_weight', 1,     30),
        'subsample':        trial.suggest_float('subsample',        0.4,   1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4,   1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha',        1e-5,  5.0,   log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda',       1e-5,  5.0,   log=True),
        'gamma':            trial.suggest_float('gamma',            0.0,   3.0),
    }
    return cv_rmse_xgb(params)

xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=N_HPO_TRIALS, show_progress_bar=True)

BEST_XGB_PARAMS = {**XGB_BASE, **xgb_study.best_params}
print(f"\nBest XGBoost  (trial #{xgb_study.best_trial.number}, "
      f"3-fold CV RMSE = {xgb_study.best_value:,.0f}):")
for k, v in xgb_study.best_params.items():
    print(f"  {k:<25} = {v}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1c — CatBoost HPO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Phase 1c — CatBoost HPO  ({N_HPO_TRIALS} trials, {N_HPO_FOLDS}-fold inner CV)")
print("═"*65)

CAT_BASE = {
    'loss_function': 'RMSE',
    'od_type':       'Iter',
    'od_wait':       100,
    'verbose':       0,
    'random_seed':   42,
    'task_type':     'CPU',
}

def cat_objective(trial):
    params = {
        **CAT_BASE,
        'learning_rate':       trial.suggest_float('learning_rate',       0.005, 0.10,  log=True),
        'depth':               trial.suggest_int  ('depth',               4,     12),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg',         1e-2,  10.0,  log=True),
        'random_strength':     trial.suggest_float('random_strength',     0.1,   5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0,   3.0),
        'border_count':        trial.suggest_int  ('border_count',        32,    255),
    }
    return cv_rmse_cat(params)

cat_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_objective, n_trials=N_HPO_TRIALS, show_progress_bar=True)

BEST_CAT_PARAMS = {**CAT_BASE, **cat_study.best_params}
print(f"\nBest CatBoost  (trial #{cat_study.best_trial.number}, "
      f"3-fold CV RMSE = {cat_study.best_value:,.0f}):")
for k, v in cat_study.best_params.items():
    print(f"  {k:<25} = {v}")

# ══════════════════════════════════════════════════════════════════════════════
# HPO SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("HPO Summary — 3-fold CV RMSE")
print("═"*65)

V14_PARAMS = {
    'lgb': {'learning_rate': 0.010021, 'num_leaves': 323, 'min_child_samples': 43,
            'feature_fraction': 0.5670, 'bagging_fraction': 0.8889,
            'reg_alpha': 0.000179, 'reg_lambda': 0.000387},
    'xgb': {'learning_rate': 0.010508, 'max_depth': 10, 'min_child_weight': 10,
            'subsample': 0.7073, 'colsample_bytree': 0.5051,
            'reg_alpha': 0.002358, 'reg_lambda': 0.10765, 'gamma': 0.2759},
    'cat': {'learning_rate': 0.031864, 'depth': 10, 'l2_leaf_reg': 1.2695,
            'random_strength': 1.5406, 'bagging_temperature': 1.2874, 'border_count': 205},
}

print(f"\n  Model       v14 params (single-split HPO)   v15 params ({N_HPO_FOLDS}-fold HPO)")
print(f"  {'─'*60}")
print(f"  LightGBM    val RMSE = (single split)        3-fold CV = {lgb_study.best_value:,.0f}")
print(f"  XGBoost     val RMSE = (single split)        3-fold CV = {xgb_study.best_value:,.0f}")
print(f"  CatBoost    val RMSE = (single split)        3-fold CV = {cat_study.best_value:,.0f}")

print(f"\n  LightGBM params changed:")
for k in lgb_study.best_params:
    old = V14_PARAMS['lgb'].get(k, 'N/A')
    new = lgb_study.best_params[k]
    flag = '  ←' if abs(new - old) / (abs(old) + 1e-10) > 0.05 else ''
    print(f"    {k:<25} {str(old):>12}  →  {str(round(new, 6)):<12}{flag}")

print(f"\n  XGBoost params changed:")
for k in xgb_study.best_params:
    old = V14_PARAMS['xgb'].get(k, 'N/A')
    new = xgb_study.best_params[k]
    flag = '  ←' if isinstance(old, float) and abs(new - old) / (abs(old) + 1e-10) > 0.05 else ''
    print(f"    {k:<25} {str(old):>12}  →  {str(round(new, 6)):<12}{flag}")

print(f"\n  CatBoost params changed:")
for k in cat_study.best_params:
    old = V14_PARAMS['cat'].get(k, 'N/A')
    new = cat_study.best_params[k]
    flag = '  ←' if isinstance(old, float) and abs(new - old) / (abs(old) + 1e-10) > 0.05 else ''
    print(f"    {k:<25} {str(old):>12}  →  {str(round(new, 6)):<12}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Final 5-fold OOF  (full leak-free pipeline from v14)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"Phase 2 — Final {N_OOF_FOLDS}-fold OOF  (leak-free encoding per fold)")
print("═"*65)

kf_oof = KFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=42)

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

X_test_base = test_fe[BASE_FEATURES]

for fold, (tr_idx, val_idx) in enumerate(kf_oof.split(np.arange(len(train_fe))), 1):
    print(f"\n  Fold {fold}/{N_OOF_FOLDS}  train={len(tr_idx):,}  val={len(val_idx):,}")

    df_tr_f  = train_fe.iloc[tr_idx]
    df_val_f = train_fe.iloc[val_idx]

    tr_enc, val_enc, te_enc = target_encode_fold(
        df_tr=df_tr_f, df_val=df_val_f, df_te=test_fe,
        cols=TARGET_ENC_COLS, target_col=TARGET,
    )

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

    m_lgb = fit_lgb(X_tr, y_tr, X_val, y_val, BEST_LGB_PARAMS, N_EST_FINAL)
    oof_lgb[val_idx] = m_lgb.predict(X_val)
    pred_lgb        += m_lgb.predict(X_te) / N_OOF_FOLDS
    imp_lgb_folds.append(m_lgb.booster_.feature_importance(importance_type='gain'))
    print_metrics(f"    LGB fold {fold}", y_val, oof_lgb[val_idx])

    m_xgb = fit_xgb(X_tr, y_tr, X_val, y_val, BEST_XGB_PARAMS, N_EST_FINAL)
    oof_xgb[val_idx] = m_xgb.predict(X_val)
    pred_xgb        += m_xgb.predict(X_te) / N_OOF_FOLDS
    xgb_scores = m_xgb.get_booster().get_score(importance_type='gain')
    imp_xgb_folds.append([xgb_scores.get(f, 0.0) for f in feature_names])
    print_metrics(f"    XGB fold {fold}", y_val, oof_xgb[val_idx])

    m_cat = fit_cat(X_tr, y_tr, X_val, y_val, BEST_CAT_PARAMS, N_EST_FINAL)
    oof_cat[val_idx] = m_cat.predict(X_val)
    pred_cat        += m_cat.predict(X_te) / N_OOF_FOLDS
    imp_cat_folds.append(m_cat.get_feature_importance())
    print_metrics(f"    CAT fold {fold}", y_val, oof_cat[val_idx])

w         = optimise_weights(y, oof_lgb, oof_xgb, oof_cat)
oof_blend = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

V14_OOF_RMSE = 21_369

print("\n" + "═"*65)
print("Final Summary")
print("═"*65)
print_metrics("LightGBM  (OOF)", y, oof_lgb)
print_metrics("XGBoost   (OOF)", y, oof_xgb)
print_metrics("CatBoost  (OOF)", y, oof_cat)
print_metrics(f"Ensemble  (L={w[0]:.2f} X={w[1]:.2f} C={w[2]:.2f})", y, oof_blend)

v15_rmse = root_mean_squared_error(y, oof_blend)
delta    = V14_OOF_RMSE - v15_rmse
print(f"\n  v14 ensemble OOF RMSE : {V14_OOF_RMSE:,}")
print(f"  v15 ensemble OOF RMSE : {v15_rmse:,.0f}  "
      f"({'↓ improved by ' + f'{delta:,.0f}' if delta > 0 else '↑ regressed by ' + f'{abs(delta):,.0f}'})")
print("═"*65)

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

pred_final = w[0]*pred_lgb + w[1]*pred_xgb + w[2]*pred_cat
pd.DataFrame({"Id": test_raw[ID_COL], "Predicted": pred_final}).to_csv(
    '../submission/submission_v15_5fold.csv', index=False
)
print("\nSubmission saved: submission_v15_5fold.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE  (5-fold average gain)
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

print(f"\nTop 30 features by mean gain (LGB / XGB / CAT):")
print(f"  {'Feature':<40} {'LGB%':>6}  {'XGB%':>6}  {'CAT%':>6}  {'Mean%':>6}")
print("  " + "-"*66)
for _, row in df_imp.head(30).iterrows():
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
