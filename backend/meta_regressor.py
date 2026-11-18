import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from meta_base import MetaBase, reg_models, EVAL_MAX_ROWS

class MetaRegressor(MetaBase):
    task, bundle_file = "reg", "meta_reg.joblib"
    def candidates(self): return reg_models()

    def evaluate(self, X, y, model_name):
        y = y.astype(float)
        if len(X) > EVAL_MAX_ROWS:
            idx = X.sample(EVAL_MAX_ROWS, random_state=0).index
            X, y = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
        pipe = self.pipe(X, model_name)
        pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
        res = y.values - pred
        mse = mean_squared_error(y, pred)
        counts, edges = np.histogram(res, bins=30)
        s = np.random.default_rng(0).choice(len(y), min(1000, len(y)), replace=False)
        return {
            "model": model_name, "validation": "5-fold cross-validation",
            "metrics": {"R2": round(r2_score(y, pred), 4), "MAE": round(mean_absolute_error(y, pred), 4),
                        "MSE": round(mse, 4), "RMSE": round(float(np.sqrt(mse)), 4)},
            "actual_vs_predicted": {"actual": y.values[s].tolist(), "predicted": pred[s].tolist()},
            "residual_hist": {"counts": counts.tolist(), "edges": edges.tolist()},
            "residuals_vs_predicted": {"predicted": pred[s].tolist(), "residual": res[s].tolist()},
            "feature_importance": self.feature_importance(pipe, X, y),
        }