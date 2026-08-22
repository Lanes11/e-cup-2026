import warnings

import catboost as cb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Constants
# =============================================================================

RANDOM_STATE = 42

GMV_COLS = [
    "gmv_search",
    "gmv_cat",
    "gmv",
]

COUNT_COLS = [
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
    "searches",
]

FLAG_COLS = [
    "search",
    "cat",
    "has_search_to_cart",
    "has_search_to_ord",
    "has_cat_to_cart",
    "has_cat_to_ord",
]


# =============================================================================
# Feature engineering
# =============================================================================


def build_features(
    df_hist: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    d = df_hist[df_hist["event_date"] <= cutoff].copy()

    # -------------------------------------------------------------------------
    # All-history aggregates
    # -------------------------------------------------------------------------

    agg_dict = {}

    for col in GMV_COLS + COUNT_COLS:
        agg_dict[f"total_{col}"] = (col, "sum")

    for col in FLAG_COLS:
        agg_dict[f"frac_{col}"] = (col, "mean")

    agg_dict["days_active"] = ("event_date", "count")
    agg_dict["first_seen"] = ("event_date", "min")
    agg_dict["last_seen"] = ("event_date", "max")
    agg_dict["max_gmv_ever"] = ("gmv", "max")

    feats = d.groupby("user_id").agg(**agg_dict).reset_index()

    feats["days_since_last"] = (cutoff - feats["last_seen"]).dt.days

    feats["days_since_first"] = (cutoff - feats["first_seen"]).dt.days + 1

    feats["active_rate"] = feats["days_active"] / feats["days_since_first"]

    feats["avg_gap"] = np.where(
        feats["days_active"] > 1,
        feats["days_since_first"] / (feats["days_active"] - 1),
        feats["days_since_first"],
    )

    feats["last_active_month"] = feats["last_seen"].dt.month
    feats["last_active_dow"] = feats["last_seen"].dt.dayofweek

    feats["returning_user"] = (feats["days_since_first"] > 90).astype(int)

    feats["active_in_last7"] = (feats["days_since_last"] <= 7).astype(int)

    # Distinct active months
    month_counts = (
        d.groupby("user_id")["event_date"]
        .apply(lambda x: x.dt.to_period("M").nunique())
        .rename("distinct_active_months")
        .reset_index()
    )

    feats = feats.merge(
        month_counts,
        on="user_id",
        how="left",
    )

    feats = feats.drop(columns=["first_seen", "last_seen"])

    # -------------------------------------------------------------------------
    # Rolling windows
    # -------------------------------------------------------------------------

    for window in [7, 14, 30, 60, 90]:
        start = cutoff - pd.Timedelta(days=window - 1)

        dw = d[d["event_date"] >= start]

        w_agg = (
            dw.groupby("user_id")
            .agg(
                **{f"w{window}_{col}": (col, "sum") for col in GMV_COLS + COUNT_COLS},
                **{
                    f"w{window}_active_days": (
                        "event_date",
                        "count",
                    )
                },
                **{
                    f"w{window}_max_gmv": (
                        "gmv",
                        "max",
                    )
                },
                **{
                    f"w{window}_std_gmv": (
                        "gmv",
                        "std",
                    )
                },
            )
            .reset_index()
        )

        feats = feats.merge(
            w_agg,
            on="user_id",
            how="left",
        )

    # -------------------------------------------------------------------------
    # Velocity: w30_gmv vs previous 30 days
    # -------------------------------------------------------------------------

    start_prev = cutoff - pd.Timedelta(days=59)
    end_prev = cutoff - pd.Timedelta(days=30)

    d_prev = d[(d["event_date"] >= start_prev) & (d["event_date"] <= end_prev)]

    prev30 = d_prev.groupby("user_id")["gmv"].sum().rename("prev30_gmv").reset_index()

    feats = feats.merge(
        prev30,
        on="user_id",
        how="left",
    )

    feats["velocity_gmv"] = (feats["w30_gmv"] - feats["prev30_gmv"]) / (
        feats["prev30_gmv"] + 1
    )

    feats = feats.drop(columns=["prev30_gmv"])

    # -------------------------------------------------------------------------
    # Trend: 30d / 90d
    # -------------------------------------------------------------------------

    for col in [
        "gmv",
        "gmv_search",
        "gmv_cat",
        "to_ord",
    ]:
        w90 = feats[f"w90_{col}"].replace(
            0,
            np.nan,
        )

        feats[f"trend_{col}_30_90"] = feats[f"w30_{col}"] / w90

    # -------------------------------------------------------------------------
    # GMV share
    # -------------------------------------------------------------------------

    total_gmv = feats["total_gmv"].replace(
        0,
        np.nan,
    )

    feats["search_gmv_share"] = feats["total_gmv_search"] / total_gmv

    feats["cat_gmv_share"] = feats["total_gmv_cat"] / total_gmv

    # -------------------------------------------------------------------------
    # Conversion rates
    # -------------------------------------------------------------------------

    conversion_pairs = [
        ("search_to_ord", "search_to_cart"),
        ("cat_to_ord", "cat_to_cart"),
        ("to_ord", "to_cart"),
    ]

    for num, den in conversion_pairs:
        denom = feats[f"total_{den}"].replace(
            0,
            np.nan,
        )

        feats[f"conv_{num}_per_{den}"] = feats[f"total_{num}"] / denom

    # -------------------------------------------------------------------------
    # GMV per order
    # -------------------------------------------------------------------------

    aov_pairs = [
        ("gmv_search", "search_to_ord"),
        ("gmv_cat", "cat_to_ord"),
        ("gmv", "to_ord"),
    ]

    for gmv_col, ord_col in aov_pairs:
        denom = feats[f"total_{ord_col}"].replace(
            0,
            np.nan,
        )

        feats[f"aov_{gmv_col}"] = feats[f"total_{gmv_col}"] / denom

    # -------------------------------------------------------------------------
    # EWM of GMV on active days only
    # -------------------------------------------------------------------------

    daily_gmv = d[
        [
            "user_id",
            "event_date",
            "gmv",
        ]
    ].sort_values(["user_id", "event_date"])

    for span in [7, 14, 30]:
        ewm_last = (
            daily_gmv.groupby("user_id")["gmv"]
            .apply(
                lambda s: (
                    s.ewm(
                        span=span,
                        adjust=False,
                    )
                    .mean()
                    .iloc[-1]
                )
            )
            .rename(f"ewm_gmv_span{span}")
            .reset_index()
        )

        feats = feats.merge(
            ewm_last,
            on="user_id",
            how="left",
        )

    feats = feats.fillna(0)

    return feats


# =============================================================================
# Load data
# =============================================================================

print("Loading data...")

df = pd.read_parquet("datasets/train_clean.parquet")

df["event_date"] = pd.to_datetime(df["event_date"])

print(f"  Shape: {df.shape}, " f"users: {df['user_id'].nunique()}")


# =============================================================================
# Dataset creation
# =============================================================================


def make_dataset(
    feat_cutoff: pd.Timestamp,
    tgt_start: pd.Timestamp,
    tgt_end: pd.Timestamp,
    label: str,
) -> tuple:
    print(f"\nBuilding [{label}] features " f"(cutoff {feat_cutoff.date()})...")

    X_raw = build_features(
        df,
        feat_cutoff,
    )

    target_df = (
        df[(df["event_date"] >= tgt_start) & (df["event_date"] <= tgt_end)]
        .groupby("user_id")
        .agg(target=("gmv", "sum"))
        .reset_index()
    )

    merged = X_raw.merge(
        target_df,
        on="user_id",
        how="left",
    )

    merged["target"] = merged["target"].fillna(0)

    feat_cols = [c for c in merged.columns if c not in ["user_id", "target"]]

    return merged, feat_cols


# =============================================================================
# Validation splits
# =============================================================================

ds, feat_cols = make_dataset(
    pd.Timestamp("2026-01-13"),
    pd.Timestamp("2026-01-14"),
    pd.Timestamp("2026-02-13"),
    "split jan→feb",
)

print(f"\nFeature count: {len(feat_cols)}")

X = ds[feat_cols].values
y = ds["target"].values
y_log = np.log1p(y)

# For tuning/final evaluation we use split2
X_tr, X_val, y_tr_log, y_val_log, y_tr, y_val = train_test_split(
    X,
    y_log,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)


# =============================================================================
# Metrics
# =============================================================================


def rmsle(
    y_true,
    y_pred_log,
):
    return np.sqrt(
        mean_squared_error(
            np.log1p(y_true),
            y_pred_log,
        )
    )


# =============================================================================
# Optuna: XGBoost
# =============================================================================

print("\nOptuna tuning XGBoost (20 trials)...")


def xgb_objective(trial):
    params = dict(
        objective="reg:squarederror",
        n_estimators=trial.suggest_int(
            "n_estimators",
            500,
            3000,
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        ),
        max_depth=trial.suggest_int(
            "max_depth",
            3,
            8,
        ),
        subsample=trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
        min_child_weight=trial.suggest_int(
            "min_child_weight",
            1,
            20,
        ),
        reg_alpha=trial.suggest_float(
            "reg_alpha",
            1e-8,
            10,
            log=True,
        ),
        reg_lambda=trial.suggest_float(
            "reg_lambda",
            1e-8,
            10,
            log=True,
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False,
    )

    return rmsle(
        y_val,
        model.predict(X_val),
    )


xgb_study = optuna.create_study(direction="minimize")

xgb_study.optimize(
    xgb_objective,
    n_trials=20,
)

best_xgb_params = xgb_study.best_params

print(f"  Best XGBoost RMSLE: " f"{xgb_study.best_value:.5f}")

print(f"  Params: {best_xgb_params}")

# =============================================================================
# Optuna: CatBoost
# =============================================================================

print("\nOptuna tuning CatBoost (20 trials)...")


def cb_objective(trial):
    params = dict(
        loss_function="RMSE",
        iterations=trial.suggest_int(
            "iterations",
            500,
            3000,
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        ),
        depth=trial.suggest_int(
            "depth",
            3,
            8,
        ),
        subsample=trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        l2_leaf_reg=trial.suggest_float(
            "l2_leaf_reg",
            1,
            10,
        ),
        min_data_in_leaf=trial.suggest_int(
            "min_data_in_leaf",
            1,
            50,
        ),
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50,
    )

    model = cb.CatBoostRegressor(**params)

    model.fit(
        X_tr,
        y_tr_log,
        eval_set=(X_val, y_val_log),
    )

    return rmsle(
        y_val,
        model.predict(X_val),
    )


cb_study = optuna.create_study(direction="minimize")

cb_study.optimize(
    cb_objective,
    n_trials=20,
)

best_cb_params = cb_study.best_params

print(f"  Best CatBoost RMSLE: " f"{cb_study.best_value:.5f}")

print(f"  Params: {best_cb_params}")

# =============================================================================
# Final models on split
# =============================================================================

print("\nTraining final models on split2 data...")

final_xgb = xgb.XGBRegressor(
    **best_xgb_params,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)

final_xgb.fit(
    X_tr,
    y_tr_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)

xgb_val_rmsle = rmsle(
    y_val,
    final_xgb.predict(X_val),
)

final_cb = cb.CatBoostRegressor(
    **best_cb_params,
    loss_function="RMSE",
    random_seed=RANDOM_STATE,
    verbose=False,
    early_stopping_rounds=50,
)

final_cb.fit(
    X_tr,
    y_tr_log,
    eval_set=(
        X_val,
        y_val_log,
    ),
)

cb_val_rmsle = rmsle(
    y_val,
    final_cb.predict(X_val),
)

ens_val_rmsle = rmsle(
    y_val,
    (final_xgb.predict(X_val) + final_cb.predict(X_val)) / 2,
)

# =============================================================================
# Submission features
# =============================================================================

print("\nBuilding submission features " "(cutoff 2026-02-13)...")

X_sub_raw = build_features(
    df,
    pd.Timestamp("2026-02-13"),
)

all_users = pd.DataFrame({"user_id": df["user_id"].unique()})

X_sub_raw = all_users.merge(
    X_sub_raw,
    on="user_id",
    how="left",
).fillna(0)

X_sub = X_sub_raw[feat_cols].values

# =============================================================================
# Retrain on all split data
# =============================================================================

xgb_best_n = final_xgb.best_iteration + 1

cb_best_n = final_cb.best_iteration_ + 1

xgb_sub_params = {k: v for k, v in best_xgb_params.items() if k != "n_estimators"}

cb_sub_params = {k: v for k, v in best_cb_params.items() if k != "iterations"}

final_xgb_full = xgb.XGBRegressor(
    **xgb_sub_params,
    n_estimators=xgb_best_n,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)

final_xgb_full.fit(
    X,
    y_log,
)

final_cb_full = cb.CatBoostRegressor(
    **cb_sub_params,
    iterations=cb_best_n,
    loss_function="RMSE",
    random_seed=RANDOM_STATE,
    verbose=False,
)

final_cb_full.fit(
    X,
    y_log,
)


# =============================================================================
# Optimize ensemble weights
# =============================================================================


def ensemble_rmsle(
    weights,
    preds_list,
    y_true,
):
    weights = np.array(weights)

    weights = weights / weights.sum()

    blended = sum(weights[i] * preds_list[i] for i in range(len(preds_list)))

    return rmsle(
        y_true,
        blended,
    )


preds_val = [
    final_xgb.predict(X_val),
    final_cb.predict(X_val),
]

result = minimize(
    ensemble_rmsle,
    x0=[0.5, 0.5],
    args=(
        preds_val,
        y_val,
    ),
    method="Nelder-Mead",
    options={
        "xatol": 1e-6,
        "fatol": 1e-6,
        "maxiter": 1000,
    },
)

raw_weights = result.x

opt_weights = raw_weights / raw_weights.sum()

print(f"Optimal weights — " f"XGB: {opt_weights[0]:.4f}, " f"CB: {opt_weights[1]:.4f}")

print(f"Ensemble RMSLE (optimized): " f"{result.fun:.5f}")

# =============================================================================
# Submission prediction
# =============================================================================

preds_sub = [
    final_xgb_full.predict(X_sub),
    final_cb_full.predict(X_sub),
]

ens_sub = sum(opt_weights[i] * preds_sub[i] for i in range(len(preds_sub)))

submission = pd.DataFrame(
    {
        "user_id": X_sub_raw["user_id"].values,
        "predict": np.clip(
            np.expm1(ens_sub),
            0,
            None,
        ),
    }
)

submission.to_csv(
    "submission_v3.csv",
    index=False,
)
