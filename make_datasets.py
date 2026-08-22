import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from features import build_features

# =============================================================================
# Load data
# =============================================================================

print("Loading data...")

df = pd.read_parquet("datasets/train_clean.parquet")
df["event_date"] = pd.to_datetime(df["event_date"])

print(f"  Shape: {df.shape}, users: {df['user_id'].nunique()}")

SPLITS = {
    "split1": {
        "feat_cutoff": pd.Timestamp("2025-12-13"),
        "tgt_start": pd.Timestamp("2025-12-14"),
        "tgt_end": pd.Timestamp("2026-01-13"),
    },
    "split2": {
        "feat_cutoff": pd.Timestamp("2026-01-13"),
        "tgt_start": pd.Timestamp("2026-01-14"),
        "tgt_end": pd.Timestamp("2026-02-13"),
    },
}

FEATURES_GROUP = ["all_history", "rolling_windows", "velocity", "trend", "gmv", "conversion_rates", "ewm"]

# =============================================================================
# Helpers
# =============================================================================


def build_target(tgt_start: pd.Timestamp, tgt_end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["event_date"] >= tgt_start) & (df["event_date"] <= tgt_end)
    return (
        df[mask]
        .groupby("user_id")
        .agg(target=("gmv", "sum"))
        .reset_index()
    )


def attach_target(feats: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    all_users = pd.DataFrame({"user_id": df["user_id"].unique()})
    merged = all_users.merge(feats, on="user_id", how="left").fillna(0)
    merged = merged.merge(target_df, on="user_id", how="left")
    merged["target"] = merged["target"].fillna(0)
    return merged


def save(frame: pd.DataFrame, split_name: str, features) -> None:
    path = f"datasets/{split_name}_{features}.parquet"
    frame.to_parquet(path, index=False)
    print(f"  Saved {path}  ({frame.shape[0]} rows, {frame.shape[1]} cols)")


# =============================================================================
# Main loop
# =============================================================================

for split_name, cfg in SPLITS.items():
    print(f"\n{'=' * 60}")
    print(f"Split: {split_name}  (feat cutoff {cfg['feat_cutoff'].date()})")
    print(f"{'=' * 60}")


    feature_flags = {name: True for name in FEATURES_GROUP}

    feats = build_features(df, cfg["feat_cutoff"], **feature_flags)
    target_df = build_target(cfg["tgt_start"], cfg["tgt_end"])
    full = attach_target(feats, target_df)

    save(full, f"{split_name}", feature_flags)


print("\nDone.")
