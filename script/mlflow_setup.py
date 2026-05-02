"""Shared MLflow configuration for HDB pipeline scripts (v1, v2, …)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd

DEFAULT_TRACKING_URI = "http://127.0.0.1:5005"
DEFAULT_EXPERIMENT_NAME = "HDB Resale Regression (Kaggle)"

# MLflow param values must stay reasonably short in some backends.
_MAX_PARAM_LEN = 500


def configure_mlflow() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI).strip()
    mlflow.set_tracking_uri(uri)
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    mlflow.set_experiment(experiment)


def log_features_json(features: list[str], artifact_filename: str = "features.json") -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / artifact_filename
        path.write_text(json.dumps(features, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_params_json(params: Mapping[str, Any], artifact_filename: str = "params.json") -> None:
    """Serialize full hyperparameter dict as a run artifact."""

    def _default(o: Any) -> Any:
        return str(o)

    with TemporaryDirectory() as td:
        path = Path(td) / artifact_filename
        path.write_text(json.dumps(dict(params), indent=2, default=_default), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_param_dict_flat(prefix: str, d: Mapping[str, Any]) -> None:
    """Log scalar-friendly entries as MLflow params; stringify the rest."""

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (bool, np.bool_)):
            mlflow.log_param(key, int(bool(v)))
        elif isinstance(v, (np.integer, int)):
            mlflow.log_param(key, int(v))
        elif isinstance(v, (np.floating, float)):
            mlflow.log_param(key, float(v))
        elif isinstance(v, str):
            mlflow.log_param(key, v[:_MAX_PARAM_LEN])
        else:
            s = json.dumps(v, default=str)
            mlflow.log_param(key, s[:_MAX_PARAM_LEN])


def log_importance_csv(
    importances: Any,
    feature_names: list[str],
    artifact_filename: str = "feature_importance.csv",
) -> None:
    df = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances, dtype=float).ravel()}
    ).sort_values("importance", ascending=False)
    with TemporaryDirectory() as td:
        path = Path(td) / artifact_filename
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def log_importance_combined_csv(
    importance_lgb: Any,
    importance_xgb: Any,
    importance_cat: Any,
    feature_names: list[str],
    artifact_filename: str = "feature_importance_combined.csv",
) -> None:
    """Log a combined feature-importance CSV for LGB / XGB / CAT.

    Each model's raw gain values are normalised to percentage (sum = 100).
    Columns: feature, lgb_pct, xgb_pct, cat_pct, mean_pct.
    Sorted descending by mean_pct.
    """

    def _to_pct(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).ravel()
        total = arr.sum()
        return 100.0 * arr / total if total > 0 else arr

    lgb_pct = _to_pct(importance_lgb)
    xgb_pct = _to_pct(importance_xgb)
    cat_pct = _to_pct(importance_cat)

    df = pd.DataFrame({
        "feature":  feature_names,
        "lgb_pct":  lgb_pct,
        "xgb_pct":  xgb_pct,
        "cat_pct":  cat_pct,
    })
    df["mean_pct"] = df[["lgb_pct", "xgb_pct", "cat_pct"]].mean(axis=1)
    df = df.sort_values("mean_pct", ascending=False).reset_index(drop=True)

    with TemporaryDirectory() as td:
        path = Path(td) / artifact_filename
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def _nested_run_name(prefix: str, base: str) -> str:
    return f"{prefix}{base}" if prefix else base


def log_nested_gbdt_holdout_triplet(
    *,
    y_val,
    val_lgb,
    val_xgb,
    val_cat,
    lgb_params: dict[str, Any],
    xgb_params: dict[str, Any],
    cat_params: dict[str, Any],
    m_lgb,
    m_xgb,
    m_cat,
    features: list[str],
    train_seconds: tuple[float, float, float],
    metrics_fn: Callable[..., tuple[float, float, float, float]],
    nested_prefix: str = "",
) -> None:
    """Log three nested runs for an 80/20-style hold-out (metrics on validation rows)."""

    sec_lgb, sec_xgb, sec_cat = train_seconds

    rmse, mae, mape, r2 = metrics_fn(y_val, val_lgb)
    with mlflow.start_run(nested=True, run_name=_nested_run_name(nested_prefix, "LightGBM")):
        log_params_json(lgb_params)
        log_param_dict_flat("lgb", lgb_params)
        mlflow.log_metrics({"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2})
        mlflow.log_metric("training_time_seconds", sec_lgb)
        log_importance_csv(m_lgb.feature_importances_, features)
        mlflow.lightgbm.log_model(m_lgb, name="model", serialization_format="skops")

    rmse, mae, mape, r2 = metrics_fn(y_val, val_xgb)
    with mlflow.start_run(nested=True, run_name=_nested_run_name(nested_prefix, "XGBoost")):
        log_params_json(xgb_params)
        log_param_dict_flat("xgb", xgb_params)
        mlflow.log_metrics({"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2})
        mlflow.log_metric("training_time_seconds", sec_xgb)
        log_importance_csv(m_xgb.feature_importances_, features)
        mlflow.xgboost.log_model(m_xgb.get_booster(), name="model")

    rmse, mae, mape, r2 = metrics_fn(y_val, val_cat)
    with mlflow.start_run(nested=True, run_name=_nested_run_name(nested_prefix, "CatBoost")):
        log_params_json(cat_params)
        log_param_dict_flat("cat", cat_params)
        mlflow.log_metrics({"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2})
        mlflow.log_metric("training_time_seconds", sec_cat)
        log_importance_csv(m_cat.get_feature_importance(), features)
        mlflow.catboost.log_model(m_cat, name="model")


def log_nested_oof_gbdt_triplet(
    *,
    y_true,
    oof_lgb: np.ndarray,
    oof_xgb: np.ndarray,
    oof_cat: np.ndarray,
    lgb_params: dict[str, Any],
    xgb_params: dict[str, Any],
    cat_params: dict[str, Any],
    m_lgb_last,
    m_xgb_last,
    m_cat_last,
    features: list[str],
    train_seconds: tuple[float, float, float],
    metrics_fn: Callable[..., tuple[float, float, float, float]],
    nested_prefix: str = "",
    importance_lgb: np.ndarray | None = None,
    importance_xgb: np.ndarray | None = None,
    importance_cat: np.ndarray | None = None,
) -> None:
    """Log three nested runs for K-fold OOF metrics; serialized models are last-fold checkpoints."""

    sec_lgb, sec_xgb, sec_cat = train_seconds
    ilgb = (
        importance_lgb
        if importance_lgb is not None
        else np.asarray(m_lgb_last.feature_importances_, dtype=float)
    )
    ixgb = (
        importance_xgb
        if importance_xgb is not None
        else np.asarray(m_xgb_last.feature_importances_, dtype=float)
    )
    icat = (
        importance_cat
        if importance_cat is not None
        else np.asarray(m_cat_last.get_feature_importance(), dtype=float)
    )

    rmse, mae, mape, r2 = metrics_fn(y_true, oof_lgb)
    with mlflow.start_run(nested=True, run_name=_nested_run_name(nested_prefix, "LightGBM")):
        log_params_json(lgb_params)
        log_param_dict_flat("lgb", lgb_params)
        mlflow.log_metrics({"oof_rmse": rmse, "oof_mae": mae, "oof_mape": mape, "oof_r2": r2})
        mlflow.log_metric("training_time_seconds", sec_lgb)
        mlflow.log_param("logged_model_note", "last_fold_checkpoint_only")
        log_importance_csv(ilgb, features)
        mlflow.lightgbm.log_model(m_lgb_last, name="model", serialization_format="skops")

    rmse, mae, mape, r2 = metrics_fn(y_true, oof_xgb)
    with mlflow.start_run(nested=True, run_name=_nested_run_name(nested_prefix, "XGBoost")):
        log_params_json(xgb_params)
        log_param_dict_flat("xgb", xgb_params)
        mlflow.log_metrics({"oof_rmse": rmse, "oof_mae": mae, "oof_mape": mape, "oof_r2": r2})
        mlflow.log_metric("training_time_seconds", sec_xgb)
        mlflow.log_param("logged_model_note", "last_fold_checkpoint_only")
        log_importance_csv(ixgb, features)
        mlflow.xgboost.log_model(m_xgb_last.get_booster(), name="model")

    rmse, mae, mape, r2 = metrics_fn(y_true, oof_cat)
    with mlflow.start_run(nested=True, run_name=_nested_run_name(nested_prefix, "CatBoost")):
        log_params_json(cat_params)
        log_param_dict_flat("cat", cat_params)
        mlflow.log_metrics({"oof_rmse": rmse, "oof_mae": mae, "oof_mape": mape, "oof_r2": r2})
        mlflow.log_metric("training_time_seconds", sec_cat)
        mlflow.log_param("logged_model_note", "last_fold_checkpoint_only")
        log_importance_csv(icat, features)
        mlflow.catboost.log_model(m_cat_last, name="model")
