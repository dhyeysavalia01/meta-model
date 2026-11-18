import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
    ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.naive_bayes import GaussianNB
from meta_features import extract_meta_features
from preprocessing import make_preprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
EVAL_MAX_ROWS = 10000

def clf_models():
    return {"LogisticRegression": LogisticRegression(max_iter=500), "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=0),
            "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=0),
            "HistGB": HistGradientBoostingClassifier(random_state=0), "NaiveBayes": GaussianNB()}

def reg_models():
    return {"Ridge": Ridge(), "Lasso": Lasso(max_iter=5000), "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=0),
            "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=0),
            "HistGB": HistGradientBoostingRegressor(random_state=0)}

class MetaBase:
    task = None            # "clf" | "reg"
    bundle_file = None

    def __init__(self):
        b = joblib.load(MODELS_DIR / self.bundle_file)
        self.model, self.features, self.classes = b["model"], b["features"], b["classes"]
        self.importance = dict(zip(self.features, self.model[-1].feature_importances_))

    def candidates(self): raise NotImplementedError

    def pipe(self, X, name):
        return Pipeline([("p", make_preprocessor(X)), ("m", self.candidates()[name])])

    def recommend(self, X_raw, y):
        mf = extract_meta_features(X_raw, y, self.task)
        row = pd.DataFrame([mf]).reindex(columns=self.features)
        proba = self.model.predict_proba(row)[0]
        order = np.argsort(-proba)
        ranking = [{"model": self.classes[i], "confidence": round(float(proba[i]), 3)} for i in order]
        top = sorted(self.features, key=lambda f: -self.importance[f])[:6]
        drivers = [{"feature": f,
                    "value": None if pd.isna(row[f].iloc[0]) else round(float(row[f].iloc[0]), 4),
                    "importance": round(float(self.importance[f]), 3)} for f in top]
        return {"recommended": ranking[0]["model"], "confidence": ranking[0]["confidence"],
                "ranking": ranking, "drivers": drivers}

    def feature_importance(self, pipe, X, y, stratify=False):
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                                  stratify=y if stratify else None)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        pipe.fit(Xtr, ytr)
        Xte, yte = Xte.iloc[:500], yte.iloc[:500]
        r = permutation_importance(pipe, Xte, yte, n_repeats=3, random_state=0, n_jobs=-1)
        s = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(10)
        return [{"feature": k, "importance": round(float(v), 4)} for k, v in s.items()]