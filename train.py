import warnings

import catboost as cb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42


# =============================================================================
# Metrics
# =============================================================================


def rmsle(y_true, y_pred_log):
    return np.sqrt(mean_squared_error(np.log1p(y_true), y_pred_log))


# =============================================================================
# Load datasets
# CatBoost trains on trend features, XGBoost on gmv features.
# split1 = train, split2 = validation.
# =============================================================================

print("Loading datasets...")


xgb_train = pd.read_parquet("datasets/split1_gmv.parquet")
xgb_val   = pd.read_parquet("datasets/split2_gmv.parquet")


def get_feat_cols(df):
    return [
        c for c in df.columns
        if c not in ("user_id", "target")
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]



xgb_feat_cols = get_feat_cols(xgb_train)


print(f"  XGBoost   — train: {xgb_train.shape[0]} rows, {len(xgb_feat_cols)} features")


X_xgb_tr      = xgb_train[xgb_feat_cols].values
y_xgb_tr      = np.log1p(xgb_train["target"].values)
X_xgb_val     = xgb_val[xgb_feat_cols].values
y_xgb_val     = np.log1p(xgb_val["target"].values)
y_xgb_val_raw = xgb_val["target"].values

# use cb_val user order as the ensemble reference
y_val_raw = y_cb_val_raw


# =============================================================================
# Optuna: XGBoost on gmv features
# =============================================================================

print("\nOptuna tuning XGBoost (20 trials)...")


def xgb_objective(trial):
    params = dict(
        objective="reg:squarederror",
        n_estimators=trial.suggest_int("n_estimators", 500, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_xgb_tr, y_xgb_tr, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
    return rmsle(y_xgb_val_raw, model.predict(X_xgb_val))


xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(xgb_objective, n_trials=20)
best_xgb_params = xgb_study.best_params

print(f"  Best XGBoost RMSLE: {xgb_study.best_value:.5f}")
print(f"  Params: {best_xgb_params}")


# =============================================================================
# Final models
# =============================================================================

print("\nTraining final models...")

final_xgb = xgb.XGBRegressor(
    **best_xgb_params,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)
final_xgb.fit(X_xgb_tr, y_xgb_tr, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)



xgb_val_rmsle = rmsle(y_xgb_val_raw, final_xgb.predict(X_xgb_val))

# align xgb predictions to cb_val user order for ensemble scoring
xgb_preds_aligned = (
    xgb_val[["user_id"]]
    .assign(pred=final_xgb.predict(X_xgb_val))
    .merge(cb_val[["user_id"]], on="user_id", how="right")["pred"]
    .fillna(0)
    .values
)

# =============================================================================
# Retrain on full data (train + val)
# =============================================================================

xgb_best_n = final_xgb.best_iteration + 1


X_xgb_full = np.concatenate([X_xgb_tr, X_xgb_val])
y_xgb_full = np.concatenate([y_xgb_tr, y_xgb_val])



xgb_sub_params = {k: v for k, v in best_xgb_params.items() if k != "n_estimators"}


final_xgb_full = xgb.XGBRegressor(
    **xgb_sub_params,
    n_estimators=xgb_best_n,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
final_xgb_full.fit(X_xgb_full, y_xgb_full)




# =============================================================================
# Submission
# =============================================================================

print("\nLoading submission features...")

sub_cb  = pd.read_parquet("datasets/submission_trend.parquet")
sub_xgb = pd.read_parquet("datasets/submission_gmv.parquet")

X_sub_cb  = sub_cb[cb_feat_cols].values
X_sub_xgb = sub_xgb[xgb_feat_cols].values

cb_sub_preds = final_cb_full.predict(X_sub_cb)
xgb_sub_preds_aligned = (
    sub_xgb[["user_id"]]
    .assign(pred=final_xgb_full.predict(X_sub_xgb))
    .merge(sub_cb[["user_id"]], on="user_id", how="right")["pred"]
    .fillna(0)
    .values
)

ens_sub = opt_weights[0] * xgb_sub_preds_aligned + opt_weights[1] * cb_sub_preds

submission = pd.DataFrame(
    {
        "user_id": sub_cb["user_id"].values,
        "predict": np.clip(np.expm1(ens_sub), 0, None),
    }
)

submission.to_csv("submission_v3.csv", index=False)
print(f"Saved submission_v3.csv  ({len(submission)} rows)")
