import numpy as np, pandas as pd
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NA_TOKENS = ["", " ", "?", "NA", "N/A", "n/a", "null", "NULL", "NaN", "nan"]
NUM_STRATEGIES = ("mean", "median", "most_frequent")
CAT_STRATEGIES = ("most_frequent", "constant")

def read_table(file_obj, filename):
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    return pd.read_csv(file_obj, na_values=NA_TOKENS, skipinitialspace=True)

def split_columns(df):
    num = df.select_dtypes("number").columns.tolist()
    return num, [c for c in df.columns if c not in num]

def inspect(df):
    num, cat = split_columns(df)
    miss = df.isna().sum()
    return {
        "n_samples": len(df), "n_features": df.shape[1],
        "numerical": num, "categorical": cat,
        "missing_total": int(miss.sum()),
        "missing_by_column": {c: int(v) for c, v in miss.items() if v},
        "columns": df.columns.tolist(),
        "preview": json.loads(df.head(10).to_json(orient="records"))
    }

def apply_imputation(df, numeric_strategy, categorical_strategy, constant="Unknown"):
    if numeric_strategy not in NUM_STRATEGIES:
        raise ValueError(f"numeric_strategy must be one of {NUM_STRATEGIES}")
    if categorical_strategy not in CAT_STRATEGIES:
        raise ValueError(f"categorical_strategy must be one of {CAT_STRATEGIES}")
    out = df.dropna(axis=1, how="all").copy()          # fully empty columns can't be imputed
    num, cat = split_columns(out)
    for c in num:
        s = out[c]
        fill = {"mean": s.mean, "median": s.median}.get(numeric_strategy, lambda: s.mode().iloc[0])()
        out[c] = s.fillna(fill)
    for c in cat:
        fill = constant if categorical_strategy == "constant" else out[c].mode().iloc[0]
        out[c] = out[c].fillna(fill)
    return out

def prepare_xy(raw, clean, target):
    if target not in clean.columns:
        raise ValueError(f"Target '{target}' not found (or it is entirely empty).")
    keep = raw[target].notna()                          # drop rows whose label was missing
    X = clean.loc[keep].drop(columns=target).reset_index(drop=True)
    X_raw = raw.loc[keep, X.columns].reset_index(drop=True)
    y = clean.loc[keep, target].reset_index(drop=True)
    return X, X_raw, y

def validate_xy(X, y, task, min_rows=30, min_class_count=5):
    errs = []
    if len(X) < min_rows: errs.append(f"Need at least {min_rows} valid rows (have {len(X)}).")
    if X.shape[1] < 1: errs.append("Need at least one feature column.")
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y): errs.append("Regression needs a numeric target.")
        elif y.nunique() < 10: errs.append("Target has fewer than 10 distinct values; this looks like classification.")
    else:
        counts = y.value_counts()
        if len(counts) < 2: errs.append("Classification needs at least 2 classes.")
        elif len(counts) > 50: errs.append(f"{len(counts)} classes; target looks continuous or ID-like.")
        elif counts.min() < min_class_count:
            errs.append(f"Smallest class has {counts.min()} samples; need at least {min_class_count}.")
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            errs.append("Target looks continuous; choose Regression instead.")
    return errs

def make_preprocessor(X):                               # used inside every model pipeline
    num = X.select_dtypes("number").columns
    cat = X.columns.difference(num)
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)), cat),
    ])