import pandas as pd
import numpy as np

# ─── 1. Загрузка ───────────────────────────────────────────────────────────────
df = pd.read_parquet("datasets/train.parquet")
print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.head(3))

# ─── 2. Базовая диагностика ────────────────────────────────────────────────────
print("\nПропуски:")
print(df.isnull().sum())
print(f"\nДубликаты: {df.duplicated().sum()}")

# ─── 3. Типы и диапазоны дат ──────────────────────────────────────────────────
df["event_date"] = pd.to_datetime(df["event_date"])
assert df["event_date"].min() >= pd.Timestamp("2025-01-01")
assert df["event_date"].max() <= pd.Timestamp("2026-02-13")

# ─── 4. Удаление полных дублей ────────────────────────────────────────────────
df = df.drop_duplicates()

# Дубли по (user_id, event_date) — одна запись в день на пользователя
# Если такие есть — агрегируем, а не удаляем произвольно
dup_mask = df.duplicated(subset=["user_id", "event_date"], keep=False)
if dup_mask.sum() > 0:
    print(f"Дублей по (user_id, event_date): {dup_mask.sum()}, агрегируем...")
    flag_cols = ["search", "cat",
                 "has_search_to_cart", "has_search_to_ord",
                 "has_cat_to_cart", "has_cat_to_ord"]
    count_cols = ["search_to_cart", "search_to_ord",
                  "cat_to_cart", "cat_to_ord",
                  "to_cart", "to_ord",
                  "gmv_search", "gmv_cat", "gmv",
                  "searches"]
    agg = {c: "max" for c in flag_cols}
    agg.update({c: "sum" for c in count_cols})
    df = df.groupby(["user_id", "event_date"], as_index=False).agg(agg)

# ─── 5. Обработка пропусков ───────────────────────────────────────────────────
# Числовые колонки: пропуск = нет активности → 0
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(0)

# ─── 6. Проверка логических ограничений ───────────────────────────────────────
flag_cols = ["search", "cat",
             "has_search_to_cart", "has_search_to_ord",
             "has_cat_to_cart", "has_cat_to_ord"]

# Флаги должны быть 0 или 1
for col in flag_cols:
    invalid = ~df[col].isin([0, 1])
    if invalid.sum():
        print(f"Некорректные значения в {col}: {invalid.sum()} строк — клипуем до [0,1]")
        df[col] = df[col].clip(0, 1).round().astype(int)

# Счётчики и GMV не могут быть отрицательными
count_cols = ["search_to_cart", "search_to_ord",
              "cat_to_cart", "cat_to_ord",
              "to_cart", "to_ord",
              "gmv_search", "gmv_cat", "gmv",
              "searches"]
for col in count_cols:
    neg = (df[col] < 0).sum()
    if neg:
        print(f"Отрицательных значений в {col}: {neg} — заменяем на 0")
        df[col] = df[col].clip(lower=0)

# ─── 7. Согласованность флагов и счётчиков ────────────────────────────────────
# Если has_search_to_cart=0, то search_to_cart должен быть 0 (и наоборот)
flag_count_pairs = [
    ("has_search_to_cart", "search_to_cart"),
    ("has_search_to_ord", "search_to_ord"),
    ("has_cat_to_cart", "cat_to_cart"),
    ("has_cat_to_ord", "cat_to_ord"),
]
for flag, cnt in flag_count_pairs:
    # Флаг=1, но счётчик=0 → выставляем флаг=0
    mask = (df[flag] == 1) & (df[cnt] == 0)
    if mask.sum():
        print(f"Коррекция: {flag}=1 при {cnt}=0 → {mask.sum()} строк")
        df.loc[mask, flag] = 0
    # Счётчик>0, но флаг=0 → выставляем флаг=1
    mask = (df[flag] == 0) & (df[cnt] > 0)
    if mask.sum():
        print(f"Коррекция: {cnt}>0 при {flag}=0 → {mask.sum()} строк")
        df.loc[mask, flag] = 1

# ─── 8. Типы данных ───────────────────────────────────────────────────────────
for col in flag_cols:
    df[col] = df[col].astype(np.int8)

int_cols = ["search_to_cart", "search_to_ord",
            "cat_to_cart", "cat_to_ord",
            "to_cart", "to_ord", "searches"]
for col in int_cols:
    df[col] = df[col].astype(np.int32)

# GMV оставляем float32
for col in ["gmv_search", "gmv_cat", "gmv"]:
    df[col] = df[col].astype(np.float32)

# ─── 9. Сортировка ────────────────────────────────────────────────────────────
df = df.sort_values(["user_id", "event_date"]).reset_index(drop=True)

# ─── 10. Итоговая диагностика ─────────────────────────────────────────────────
print(f"\nИтоговый shape: {df.shape}")
print(f"Уникальных пользователей: {df['user_id'].nunique()}")
print(f"Диапазон дат: {df['event_date'].min()} – {df['event_date'].max()}")
print(f"Пропуски: {df.isnull().sum().sum()}")
print(df.dtypes)

# ─── 11. Сохранение ───────────────────────────────────────────────────────────
df.to_parquet("train_clean.parquet", index=False)
print("\nСохранено в train_clean.parquet")
