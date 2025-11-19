"""
Custom sklearn transformers for data preprocessing.
Based on the custom transformers from the notebooks.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import pandas as pd


class DeleteNanRows(BaseEstimator, TransformerMixin):
    """
    Transformer to remove rows with NaN values.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.dropna()


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler using RobustScaler for specified attributes.
    """
    def __init__(self, attributes):
        self.attributes = attributes
        self.scaler = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(
            X_scaled, 
            columns=self.attributes, 
            index=X_copy.index
        )
        
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        
        return X_copy


class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    """
    Custom One-Hot Encoder for categorical features.
    """
    def __init__(self):
        self._oh = OneHotEncoder(sparse=False)
        self._columns = None
    
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(
            X_cat_oh, 
            columns=self._columns, 
            index=X_copy.index
        )
        
        X_copy = X_copy.drop(list(X_cat.columns), axis=1)
        return X_copy.join(X_cat_oh)
