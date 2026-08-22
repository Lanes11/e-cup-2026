import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

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


def build_features(
    df_hist: pd.DataFrame,
    cutoff: pd.Timestamp,
    all_history: bool = False,
    rolling_windows: bool = False,
    velocity: bool = False,
    trend: bool = False,
    gmv: bool = False,
    conversion_rates: bool = False,
    ewm: bool = False,
) -> pd.DataFrame:
    if velocity or trend:
        rolling_windows = True

    d = df_hist[df_hist["event_date"] <= cutoff].copy()

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

    # -------------------------------------------------------------------------
    # All-history aggregates
    # -------------------------------------------------------------------------
    if all_history:
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

        month_counts = (
            d.groupby("user_id")["event_date"]
            .apply(lambda x: x.dt.to_period("M").nunique())
            .rename("distinct_active_months")
            .reset_index()
        )
        feats = feats.merge(month_counts, on="user_id", how="left")
        feats = feats.drop(columns=["first_seen", "last_seen"])

    # -------------------------------------------------------------------------
    # Rolling windows
    # -------------------------------------------------------------------------
    if rolling_windows:
        for window in [7, 14, 30, 60, 90]:
            start = cutoff - pd.Timedelta(days=window - 1)
            dw = d[d["event_date"] >= start]
            w_agg = (
                dw.groupby("user_id")
                .agg(
                    **{
                        f"w{window}_{col}": (col, "sum")
                        for col in GMV_COLS + COUNT_COLS
                    },
                    **{f"w{window}_active_days": ("event_date", "count")},
                    **{f"w{window}_max_gmv": ("gmv", "max")},
                    **{f"w{window}_std_gmv": ("gmv", "std")},
                )
                .reset_index()
            )
            feats = feats.merge(w_agg, on="user_id", how="left")

    # -------------------------------------------------------------------------
    # Velocity: w30_gmv vs previous 30 days
    # -------------------------------------------------------------------------
    if velocity:
        start_prev = cutoff - pd.Timedelta(days=59)
        end_prev = cutoff - pd.Timedelta(days=30)
        d_prev = d[(d["event_date"] >= start_prev) & (d["event_date"] <= end_prev)]
        prev30 = (
            d_prev.groupby("user_id")["gmv"].sum().rename("prev30_gmv").reset_index()
        )
        feats = feats.merge(prev30, on="user_id", how="left")
        feats["velocity_gmv"] = (feats["w30_gmv"] - feats["prev30_gmv"]) / (
            feats["prev30_gmv"] + 1
        )
        feats = feats.drop(columns=["prev30_gmv"])

    # -------------------------------------------------------------------------
    # Trend: 30d / 90d
    # -------------------------------------------------------------------------
    if trend:
        for col in ["gmv", "gmv_search", "gmv_cat", "to_ord"]:
            w90 = feats[f"w90_{col}"].replace(0, np.nan)
            feats[f"trend_{col}_30_90"] = feats[f"w30_{col}"] / w90

    # -------------------------------------------------------------------------
    # GMV share / GMV per order (AOV)
    # -------------------------------------------------------------------------
    if gmv:
        total_gmv = feats["total_gmv"].replace(0, np.nan)
        feats["search_gmv_share"] = feats["total_gmv_search"] / total_gmv
        feats["cat_gmv_share"] = feats["total_gmv_cat"] / total_gmv

        for gmv_col, ord_col in [
            ("gmv_search", "search_to_ord"),
            ("gmv_cat", "cat_to_ord"),
            ("gmv", "to_ord"),
        ]:
            denom = feats[f"total_{ord_col}"].replace(0, np.nan)
            feats[f"aov_{gmv_col}"] = feats[f"total_{gmv_col}"] / denom

    # -------------------------------------------------------------------------
    # Conversion rates
    # -------------------------------------------------------------------------
    if conversion_rates:
        for num, den in [
            ("search_to_ord", "search_to_cart"),
            ("cat_to_ord", "cat_to_cart"),
            ("to_ord", "to_cart"),
        ]:
            denom = feats[f"total_{den}"].replace(0, np.nan)
            feats[f"conv_{num}_per_{den}"] = feats[f"total_{num}"] / denom

    # -------------------------------------------------------------------------
    # EWM of GMV on active days only
    # -------------------------------------------------------------------------
    if ewm:
        daily_gmv = d[["user_id", "event_date", "gmv"]].sort_values(
            ["user_id", "event_date"]
        )

        for span in [7, 14, 30]:
            ewm_last = (
                daily_gmv.groupby("user_id")["gmv"]
                .apply(lambda s: s.ewm(span=span, adjust=False).mean().iloc[-1])
                .rename(f"ewm_gmv_span{span}")
                .reset_index()
            )
            feats = feats.merge(ewm_last, on="user_id", how="left")

        feats = feats.fillna(0)

    return feats
