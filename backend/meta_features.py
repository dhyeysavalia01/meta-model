import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from preprocessing import make_preprocessor

MAX_ROWS = 5000

def landmarkers(X, y, task):
    lm = {"clf": {"lm_stump": DecisionTreeClassifier(max_depth=1),
                  "lm_1nn": KNeighborsClassifier(n_neighbors=1)},
          "reg": {"lm_stump": DecisionTreeRegressor(max_depth=1),
                  "lm_linear": LinearRegression()}}[task]
    scoring = "balanced_accuracy" if task == "clf" else "r2"
    out = {}
    for k, m in lm.items():
        try:
            out[k] = cross_val_score(Pipeline([("p", make_preprocessor(X)), ("m", m)]),
                                     X, y, cv=3, scoring=scoring).mean()
        except Exception:
            out[k] = np.nan                              # meta-model imputes it
    return out

def extract_meta_features(X, y, task):
    if len(X) > MAX_ROWS:
        idx = X.sample(MAX_ROWS, random_state=0).index
        X, y = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
    num = X.select_dtypes("number")
    n, d = X.shape
    f = {"n_samples": n, "log_n_samples": np.log(n), "n_features": d,
         "n_num": num.shape[1], "pct_cat": (d - num.shape[1]) / d,
         "pct_missing": X.isna().mean().mean(), "dim_ratio": d / n}
    if num.shape[1]:
        f["mean_skew"], f["mean_kurt"] = num.skew().mean(), num.kurt().mean()
        c = num.corr().abs().values
        f["mean_abs_corr"] = c[np.triu_indices_from(c, 1)].mean() if num.shape[1] > 1 else 0
    if task == "clf":
        p = y.value_counts(normalize=True).values
        f.update(n_classes=len(p), class_entropy=-(p * np.log(p)).sum(),
                 imbalance_ratio=p.max() / p.min())
    else:
        f.update(target_skew=y.skew(), target_kurt=y.kurt(), target_cv=y.std() / (abs(y.mean()) + 1e-9))
    f.update(landmarkers(X, y, task))
    return f