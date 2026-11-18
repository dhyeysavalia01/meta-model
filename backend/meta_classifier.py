import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from meta_base import MetaBase, clf_models, EVAL_MAX_ROWS

def _thin(fpr, tpr, n=100):
    i = np.unique(np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int))
    return {"fpr": fpr[i].round(4).tolist(), "tpr": tpr[i].round(4).tolist()}

class MetaClassifier(MetaBase):
    task, bundle_file = "clf", "meta_clf.joblib"
    def candidates(self): return clf_models()

    def evaluate(self, X, y, model_name):
        le = LabelEncoder()
        yi = pd.Series(le.fit_transform(y))
        names = le.classes_.astype(str)
        if len(X) > EVAL_MAX_ROWS:
            X, _, yi, _ = train_test_split(X, yi, train_size=EVAL_MAX_ROWS, stratify=yi, random_state=0)
            X, yi = X.reset_index(drop=True), yi.reset_index(drop=True)
        pipe = self.pipe(X, model_name)
        k = int(max(2, min(5, np.bincount(yi).min())))
        proba = cross_val_predict(pipe, X, yi, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                                  method="predict_proba")
        pred = proba.argmax(1)
        if len(names) == 2:
            auc = roc_auc_score(yi, proba[:, 1])
            roc = [{"class": str(names[1]), **_thin(*roc_curve(yi, proba[:, 1])[:2])}]
        else:
            auc = roc_auc_score(yi, proba, multi_class="ovr", average="macro")
            roc = [{"class": str(names[i]), **_thin(*roc_curve((yi == i).astype(int), proba[:, i])[:2])}
                   for i in range(min(len(names), 10))]
        return {
            "model": model_name, "validation": f"{k}-fold stratified cross-validation",
            "classes": names.tolist(),
            "confusion_matrix": confusion_matrix(yi, pred).tolist(),
            "report": classification_report(yi, pred, target_names=names, output_dict=True, zero_division=0),
            "auc": round(float(auc), 4), "auc_type": "binary" if len(names) == 2 else "macro one-vs-rest",
            "roc": roc,
            "class_distribution": dict(zip(names.tolist(), np.bincount(yi).tolist())),
            "feature_importance": self.feature_importance(pipe, X, yi, stratify=True),
        }