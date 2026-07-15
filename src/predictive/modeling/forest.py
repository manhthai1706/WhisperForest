import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional, Union

class ForestModel:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent
        self.model = None
        self.model_type = None

    def fit(self, model_type: str = "regressor", **kwargs):
        """
        Fit standard Random Forest on the dataset.
        """
        data = self.parent.data
        target = self.parent.target
        features = self.parent.features
        treatment = self.parent.treatment
        
        cols = features.copy()
        if treatment and treatment not in cols:
            cols.append(treatment)
            
        X = data[cols]
        y = data[target]
        
        self.model_type = model_type.lower()
        
        if self.model_type == "regressor":
            self.model = RandomForestRegressor(random_state=42, **kwargs)
        elif self.model_type == "classifier":
            self.model = RandomForestClassifier(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'regressor' or 'classifier'.")
            
        self.model.fit(X, y)
        print(f"RandomForest {self.model_type} trained successfully.")
        return self.model

    def predict(self, X: Union[pd.DataFrame, pd.Series]):
        """
        Predict target value using the trained forest.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Run fit() first.")
            
        features = self.parent.features
        treatment = self.parent.treatment
        cols = features.copy()
        if treatment and treatment not in cols:
            cols.append(treatment)
            
        if isinstance(X, pd.DataFrame):
            X_input = X[cols]
        else:
            X_input = pd.DataFrame([X])[cols]
            
        return self.model.predict(X_input)

    def predict_proba(self, X: Union[pd.DataFrame, pd.Series]):
        """
        Predict class probabilities (only for classifier).
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Run fit() first.")
        if not isinstance(self.model, RandomForestClassifier):
            raise ValueError("predict_proba is only available for 'classifier' model type.")
            
        features = self.parent.features
        treatment = self.parent.treatment
        cols = features.copy()
        if treatment and treatment not in cols:
            cols.append(treatment)
            
        if isinstance(X, pd.DataFrame):
            X_input = X[cols]
        else:
            X_input = pd.DataFrame([X])[cols]
            
        return self.model.predict_proba(X_input)
