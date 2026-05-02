"""
HDB Resale Price Prediction Pipeline — v20
Models   : LightGBM · XGBoost · CatBoost (v19 Optuna params, hardcoded)
Ensemble : Nelder-Mead weight optimisation on OOF predictions
Output   : submission_v20_5fold.csv

Changes vs v19
──────────────
1. Params hardcoded from v19 Optuna run — no HPO re-run.

2. Feature importance prints ALL features sorted by mean gain (LGB / XGB / CAT),
   no top-30 cutoff.

3. Raw distance columns (mrt_nearest_distance, Mall_Nearest_Distance,
   Hawker_Nearest_Distance) dropped — log versions kept instead.
"""

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

import mlflow_setup as mls

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_OOF_FOLDS   = 5     # outer OOF folds for final evaluation
EARLY_STOP    = 100
N_EST_FINAL   = 5000  # cap for final OOF (early stopping determines actual count)

TARGET_ENC_COLS = ['town', 'planning_area']
LABEL_ENC_COLS  = ['flat_type', 'flat_model', 'mrt_name', 'pri_sch_name', 'sec_sch_name']
REDUNDANT_COLS  = ['floor_area_sqft', 'hdb_age', 'lower', 'upper', 'mid',
                   'mrt_nearest_distance', 'Mall_Nearest_Distance', 'Hawker_Nearest_Distance',
                   'bus_stop_nearest_distance', 'pri_sch_nearest_distance', 'sec_sch_nearest_dist']
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

def metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2


def print_metrics(label, y_true, y_pred):
    rmse, mae, mape, r2 = metrics(y_true, y_pred)
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
    for r in radii_m:
        feat = f'spatial_{r}m_psm'
        tr_enc[feat]  = _encode(coords_tr,  psm_tr, global_psm, r)
        val_enc[feat] = _encode(coords_val, psm_tr, global_psm, r)
        te_enc[feat]  = _encode(coords_te,  psm_tr, global_psm, r)
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
    df['log_bus_dist']          = np.log1p(df['bus_stop_nearest_distance'])
    df['log_pri_sch_dist']      = np.log1p(df['pri_sch_nearest_distance'])
    df['log_sec_sch_dist']      = np.log1p(df['sec_sch_nearest_dist'])
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
SPATIAL_FEATS = [f'spatial_{r}m_te' for r in RADII_M] + [f'spatial_{r}m_psm' for r in RADII_M]

print(f"\nBase features                : {len(BASE_FEATURES)}")
print(f"Target-encoded (per fold)   : {TARGET_ENC_COLS}")
print(f"Spatial-encoded (per fold)  : {SPATIAL_FEATS}")
print(f"Total features per fold     : {len(BASE_FEATURES) + len(TARGET_ENC_COLS) + len(SPATIAL_FEATS)}")

train_fe = train_fe.reset_index(drop=True)
test_fe  = test_fe.reset_index(drop=True)
y        = train_fe[TARGET]

# ══════════════════════════════════════════════════════════════════════════════
# BEST PARAMS  (hardcoded from v19 Optuna run — trial #48 LGB, trial #35 CAT)
# ══════════════════════════════════════════════════════════════════════════════

BEST_LGB_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      42,
    'bagging_freq':      1,
    'learning_rate':     0.014089790573397976,
    'num_leaves':        230,
    'min_child_samples': 118,
    'feature_fraction':  0.40375527661427363,
    'bagging_fraction':  0.9777696606617974,
    'reg_alpha':         0.0556273239722144,
    'reg_lambda':        0.0004285252521224354,
}

BEST_XGB_PARAMS = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
    'learning_rate':    0.010741153512216828,
    'max_depth':        11,
    'min_child_weight': 29,
    'subsample':        0.6939185945889886,
    'colsample_bytree': 0.40265318949690715,
    'reg_alpha':        0.009156494872927058,
    'reg_lambda':       0.011320917796794635,
    'gamma':            2.6623939169044766,
}

BEST_CAT_PARAMS = {
    'loss_function':      'RMSE',
    'od_type':            'Iter',
    'od_wait':            100,
    'verbose':            0,
    'random_seed':        42,
    'task_type':          'CPU',
    'learning_rate':      0.03689723463971788,
    'depth':              9,
    'l2_leaf_reg':        0.04672450305870539,
    'random_strength':    0.7295014196914611,
    'bagging_temperature': 0.5812566288558311,
    'border_count':       192,
}

print("\n" + "═"*65)
print("Params — hardcoded from v19 Optuna (3-fold CV RMSE: LGB=21,777  XGB=21,808  CAT=21,793)")
print("═"*65)
print("\n  LightGBM:")
for k, v in BEST_LGB_PARAMS.items():
    print(f"    {k:<25} = {v}")
print("\n  XGBoost:")
for k, v in BEST_XGB_PARAMS.items():
    print(f"    {k:<25} = {v}")
print("\n  CatBoost:")
for k, v in BEST_CAT_PARAMS.items():
    print(f"    {k:<25} = {v}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Final 5-fold OOF + MLflow (full leak-free pipeline from v14)
# ══════════════════════════════════════════════════════════════════════════════

mls.configure_mlflow()

with mlflow.start_run(run_name="hdb_ml_pipeline_v20"):
    mlflow.set_tags({
        "script_version": "v20",
        "pipeline": "hdb_ml_pipeline",
        "cv_scheme": "5fold_oof_frozen_hyperparams",
        "hyperparams_source": "v19_optuna_hardcoded",
        "feature_importance_agg": "mean_gain_across_folds",
    })
    mlflow.log_params({
        "n_samples": len(train_fe),
        "n_base_features": len(BASE_FEATURES),
        "n_cat_te_cols": len(TARGET_ENC_COLS),
        "n_spatial_te_cols": len(SPATIAL_FEATS),
        "n_low_imp_dropped": len(LOW_IMP_COLS),
        "oof_folds": N_OOF_FOLDS,
        "n_estimators_final_cap": N_EST_FINAL,
    })
    mls.log_params_json(BEST_LGB_PARAMS, artifact_filename="lgb_best_params.json")
    mls.log_params_json(BEST_XGB_PARAMS, artifact_filename="xgb_best_params.json")
    mls.log_params_json(BEST_CAT_PARAMS, artifact_filename="cat_best_params.json")

    pipeline_t0 = time.perf_counter()

    print("\n" + "═"*65)
    print(f"Final {N_OOF_FOLDS}-fold OOF  (leak-free encoding per fold)")
    print("═"*65)

    sec_lgb = sec_xgb = sec_cat = 0.0
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

        df_tr_f = train_fe.iloc[tr_idx]
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

        tr_enc = {**tr_enc, **sp_tr}
        val_enc = {**val_enc, **sp_val}
        te_enc = {**te_enc, **sp_te}

        X_tr = attach_encodings(df_tr_f[BASE_FEATURES], tr_enc)
        X_val = attach_encodings(df_val_f[BASE_FEATURES], val_enc)
        X_te = attach_encodings(X_test_base, te_enc)

        if feature_names is None:
            feature_names = list(X_tr.columns)
            mls.log_features_json(feature_names)

        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        t0 = time.perf_counter()
        m_lgb = fit_lgb(X_tr, y_tr, X_val, y_val, BEST_LGB_PARAMS, N_EST_FINAL)
        sec_lgb += time.perf_counter() - t0
        oof_lgb[val_idx] = m_lgb.predict(X_val)
        pred_lgb += m_lgb.predict(X_te) / N_OOF_FOLDS
        imp_lgb_folds.append(m_lgb.booster_.feature_importance(importance_type='gain'))
        print_metrics(f"    LGB fold {fold}", y_val, oof_lgb[val_idx])

        t0 = time.perf_counter()
        m_xgb = fit_xgb(X_tr, y_tr, X_val, y_val, BEST_XGB_PARAMS, N_EST_FINAL)
        sec_xgb += time.perf_counter() - t0
        oof_xgb[val_idx] = m_xgb.predict(X_val)
        pred_xgb += m_xgb.predict(X_te) / N_OOF_FOLDS
        xgb_scores = m_xgb.get_booster().get_score(importance_type='gain')
        imp_xgb_folds.append([xgb_scores.get(f, 0.0) for f in feature_names])
        print_metrics(f"    XGB fold {fold}", y_val, oof_xgb[val_idx])

        t0 = time.perf_counter()
        m_cat = fit_cat(X_tr, y_tr, X_val, y_val, BEST_CAT_PARAMS, N_EST_FINAL)
        sec_cat += time.perf_counter() - t0
        oof_cat[val_idx] = m_cat.predict(X_val)
        pred_cat += m_cat.predict(X_te) / N_OOF_FOLDS
        imp_cat_folds.append(m_cat.get_feature_importance())
        print_metrics(f"    CAT fold {fold}", y_val, oof_cat[val_idx])

    ensemble_t0 = time.perf_counter()
    w = optimise_weights(y, oof_lgb, oof_xgb, oof_cat)
    mlflow.log_metric("ensemble_optimize_seconds", time.perf_counter() - ensemble_t0)

    oof_blend = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat

    erm, emae, emape, er2 = metrics(y, oof_blend)
    mlflow.log_metrics({
        "oof_ensemble_rmse": erm,
        "oof_ensemble_mae": emae,
        "oof_ensemble_mape": emape,
        "oof_ensemble_r2": er2,
        "oof_rmse_lightgbm": metrics(y, oof_lgb)[0],
        "oof_rmse_xgboost": metrics(y, oof_xgb)[0],
        "oof_rmse_catboost": metrics(y, oof_cat)[0],
        "oof_training_seconds": time.perf_counter() - pipeline_t0,
    })
    mlflow.log_param("ensemble_weight_lgb", float(w[0]))
    mlflow.log_param("ensemble_weight_xgb", float(w[1]))
    mlflow.log_param("ensemble_weight_cat", float(w[2]))

    imp_lgb_mean = np.array(imp_lgb_folds).mean(axis=0)
    imp_xgb_mean = np.array(imp_xgb_folds).mean(axis=0)
    imp_cat_mean = np.array(imp_cat_folds).mean(axis=0)

    mls.log_nested_oof_gbdt_triplet(
        y_true=y,
        oof_lgb=oof_lgb,
        oof_xgb=oof_xgb,
        oof_cat=oof_cat,
        lgb_params=BEST_LGB_PARAMS,
        xgb_params=BEST_XGB_PARAMS,
        cat_params=BEST_CAT_PARAMS,
        m_lgb_last=m_lgb,
        m_xgb_last=m_xgb,
        m_cat_last=m_cat,
        features=feature_names,
        train_seconds=(sec_lgb, sec_xgb, sec_cat),
        metrics_fn=metrics,
        importance_lgb=imp_lgb_mean,
        importance_xgb=imp_xgb_mean,
        importance_cat=imp_cat_mean,
    )

    V19_OOF_RMSE = 21_320

    print("\n" + "═"*65)
    print("Final Summary")
    print("═"*65)
    print_metrics("LightGBM  (OOF)", y, oof_lgb)
    print_metrics("XGBoost   (OOF)", y, oof_xgb)
    print_metrics("CatBoost  (OOF)", y, oof_cat)
    print_metrics(f"Ensemble  (L={w[0]:.2f} X={w[1]:.2f} C={w[2]:.2f})", y, oof_blend)

    v20_rmse = root_mean_squared_error(y, oof_blend)
    delta = V19_OOF_RMSE - v20_rmse
    print(f"\n  v19 ensemble OOF RMSE : {V19_OOF_RMSE:,}")
    print(f"  v20 ensemble OOF RMSE : {v20_rmse:,.0f}  "
          f"({'↓ improved by ' + f'{delta:,.0f}' if delta > 0 else '↑ regressed by ' + f'{abs(delta):,.0f}'})")
    print("═"*65)

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

pred_final = w[0]*pred_lgb + w[1]*pred_xgb + w[2]*pred_cat
pd.DataFrame({"Id": test_raw[ID_COL], "Predicted": pred_final}).to_csv(
    '../submission/submission_v20_5fold.csv', index=False
)
print("\nSubmission saved: submission_v20_5fold.csv")

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

print(f"\nAll features by mean gain (LGB / XGB / CAT):")
print(f"  {'Feature':<40} {'LGB%':>6}  {'XGB%':>6}  {'CAT%':>6}  {'Mean%':>6}")
print("  " + "-"*66)
for _, row in df_imp.iterrows():
    print(f"  {row['feature']:<40} {row['lgb_%']:>6.2f}  {row['xgb_%']:>6.2f}  {row['cat_%']:>6.2f}  {row['mean_%']:>6.2f}")
