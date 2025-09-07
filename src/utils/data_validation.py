import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pydantic import BaseModel, Field, validator

class DriverFeaturesSchema(BaseModel):
    """Schema for driver features validation"""
    driver_id: str = Field(..., min_length=1, max_length=50)
    speed_variance: float = Field(..., ge=0, le=100)
    harsh_acceleration_events: int = Field(..., ge=0, le=1000)
    harsh_braking_events: int = Field(..., ge=0, le=1000)
    night_driving_hours: float = Field(..., ge=0, le=744)  # Max hours per month
    weekend_driving_ratio: float = Field(..., ge=0, le=1)
    avg_trip_distance: float = Field(..., ge=0, le=1000)
    experience_years: Optional[float] = Field(None, ge=0, le=80)
    age: Optional[int] = Field(None, ge=16, le=100)
    
    @validator('weekend_driving_ratio')
    def validate_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Ratio must be between 0 and 1')
        return v

class SurvivalDataSchema(BaseModel):
    """Schema for survival analysis data validation"""
    duration: float = Field(..., gt=0, description="Time to event (positive)")
    event: int = Field(..., ge=0, le=1, description="Event indicator (0 or 1)")
    
    @validator('duration')
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        return v

class DataValidator:
    """Comprehensive data validation for driver analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_driver_features(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate driver features data"""
        errors = []
        
        try:
            DriverFeaturesSchema(**data)
            return True, []
        except Exception as e:
            errors.append(str(e))
            return False, errors
    
    def validate_survival_data(self, df: pd.DataFrame, 
                             duration_col: str, 
                             event_col: str) -> Tuple[bool, List[str]]:
        """Validate survival analysis data"""
        errors = []
        
        # Check required columns exist
        if duration_col not in df.columns:
            errors.append(f"Duration column '{duration_col}' not found")
        if event_col not in df.columns:
            errors.append(f"Event column '{event_col}' not found")
            
        if errors:
            return False, errors
        
        # Validate duration values
        if df[duration_col].isnull().any():
            errors.append("Duration column contains null values")
        if (df[duration_col] <= 0).any():
            errors.append("Duration column contains non-positive values")
            
        # Validate event values
        if df[event_col].isnull().any():
            errors.append("Event column contains null values")
        if not df[event_col].isin([0, 1]).all():
            errors.append("Event column contains values other than 0 or 1")
        
        # Check minimum sample size
        if len(df) < 10:
            errors.append("Insufficient sample size (minimum 10 required)")
            
        # Check event rate
        event_rate = df[event_col].mean()
        if event_rate < 0.05:
            errors.append(f"Very low event rate ({event_rate:.1%}), may cause convergence issues")
        elif event_rate > 0.95:
            errors.append(f"Very high event rate ({event_rate:.1%}), consider reformulating problem")
        
        return len(errors) == 0, errors
    
    def validate_feature_matrix(self, X: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate feature matrix for modeling"""
        errors = []
        
        # Check for empty dataframe
        if X.empty:
            errors.append("Feature matrix is empty")
            return False, errors
        
        # Check for missing values
        missing_cols = X.isnull().sum()
        high_missing = missing_cols[missing_cols > 0.5 * len(X)]
        if not high_missing.empty:
            errors.append(f"High missing value rate in columns: {high_missing.index.tolist()}")
        
        # Check for constant columns
        constant_cols = []
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                if X[col].nunique() == 1:
                    constant_cols.append(col)
        
        if constant_cols:
            errors.append(f"Constant columns detected: {constant_cols}")
        
        # Check for highly correlated features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )
            
            if high_corr_pairs:
                errors.append(f"Highly correlated feature pairs: {high_corr_pairs}")
        
        # Check value ranges
        for col in numeric_cols:
            col_data = X[col].dropna()
            if len(col_data) > 0:
                # Check for extreme outliers
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                if outliers > 0.1 * len(col_data):
                    errors.append(f"High outlier rate in column '{col}': {outliers} outliers")
        
        return len(errors) == 0, errors
    
    def validate_model_predictions(self, predictions: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate model prediction outputs"""
        errors = []
        
        # Check for NaN or infinite values
        if np.isnan(predictions).any():
            errors.append("Predictions contain NaN values")
        if np.isinf(predictions).any():
            errors.append("Predictions contain infinite values")
        
        # Check prediction range (assuming risk scores 0-1)
        if predictions.min() < 0 or predictions.max() > 1:
            errors.append(f"Predictions outside valid range [0,1]: min={predictions.min():.3f}, max={predictions.max():.3f}")
        
        # Check for constant predictions
        if predictions.std() < 1e-6:
            errors.append("All predictions are essentially constant")
        
        return len(errors) == 0, errors
