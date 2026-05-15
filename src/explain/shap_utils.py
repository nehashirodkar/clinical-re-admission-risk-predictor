"""SHAP TreeExplainer utilities for global and per-patient explanations.

The trained object is an imblearn Pipeline: pre -> smote -> clf.
At inference SMOTE is a no-op (it only resamples during fit), so we
transform with the fitted preprocessor and explain the XGBoost model
on the transformed feature space.
"""
import numpy as np
import shap


def _split_pipeline(pipe):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    return pre, clf


def get_feature_names(pipe) -> list[str]:
    pre, _ = _split_pipeline(pipe)
    return list(pre.get_feature_names_out())


def make_explainer(pipe) -> shap.TreeExplainer:
    _, clf = _split_pipeline(pipe)
    return shap.TreeExplainer(clf)


def explain_global(pipe, X, max_samples: int = 2000):
    pre, _ = _split_pipeline(pipe)
    Xt = pre.transform(X[:max_samples])
    explainer = make_explainer(pipe)
    shap_values = explainer.shap_values(Xt)
    names = get_feature_names(pipe)
    importance = np.abs(shap_values).mean(axis=0)
    return sorted(zip(names, importance), key=lambda t: -t[1])


def explain_one(pipe, x_row, top_k: int = 5) -> list[dict]:
    """Return the top-k risk drivers for a single patient row (DataFrame)."""
    pre, _ = _split_pipeline(pipe)
    xt = pre.transform(x_row)
    explainer = make_explainer(pipe)
    sv = explainer.shap_values(xt)[0]
    names = get_feature_names(pipe)

    order = np.argsort(-np.abs(sv))[:top_k]
    return [
        {
            "feature": names[i],
            "shap_value": float(sv[i]),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk",
        }
        for i in order
    ]
