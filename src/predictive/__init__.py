from .modeling.forest import ForestModel
from .explanation.shapley import ShapleyExplainer

class PredictiveBranch:
    def __init__(self, parent):
        self.parent = parent
        self.modeling = ForestModel(self)
        self.explanation = ShapleyExplainer(self)

    def fit(self, model_type="regressor", **kwargs):
        """
        Fit the standard RandomForest model.
        """
        self.modeling.fit(model_type, **kwargs)

    def predict(self, X):
        """
        Predict target variable.
        """
        return self.modeling.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities (only for classifier).
        """
        return self.modeling.predict_proba(X)

    def explain(self, method="shap", **kwargs):
        """
        Explain the model using SHAP or feature importances.
        """
        return self.explanation.explain(method, **kwargs)

