import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings

class CoxProportionalHazardsModel:
    """
    Cox Proportional Hazards model for driver risk analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame, duration_col: str, 
                    event_col: str, feature_cols: List[str]) -> pd.DataFrame:
        """
        Prepare data for Cox regression analysis
        
        Args:
            df: Input dataframe
            duration_col: Time to event column
            event_col: Event indicator (1=event, 0=censored)
            feature_cols: List of feature column names
            
        Returns:
            Prepared dataframe for Cox modeling
        """
        
        # Create a copy for processing
        df_processed = df.copy()
        
        # Validate required columns
        required_cols = [duration_col, event_col] + feature_cols
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed, feature_cols)
        
        # Feature engineering
        df_processed = self._engineer_survival_features(df_processed, feature_cols)
        
        # Remove records with invalid durations
        df_processed = df_processed[df_processed[duration_col] > 0]
        
        # Store feature names for later use
        self.feature_names = [col for col in df_processed.columns 
                             if col not in [duration_col, event_col]]
        
        self.logger.info(f"Prepared {len(df_processed)} records with {len(self.feature_names)} features")
        
        return df_processed
    
    def fit(self, df: pd.DataFrame, duration_col: str, event_col: str,
           penalizer: float = 0.1, l1_ratio: float = 0.0) -> Dict[str, Any]:
        """
        Fit Cox proportional hazards model
        
        Args:
            df: Prepared dataframe
            duration_col: Duration column name
            event_col: Event column name
            penalizer: Regularization strength
            l1_ratio: Elastic net mixing parameter (0=Ridge, 1=Lasso)
        """
        
        # Initialize and fit the model
        self.model = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            baseline_estimation_method="breslow"
        )
        
        try:
            # Fit the model
            self.model.fit(df, duration_col=duration_col, event_col=event_col)
            
            # Model evaluation
            results = self._evaluate_model(df, duration_col, event_col)
            
            # Proportional hazards assumption testing
            ph_test = self._test_proportional_hazards(df, duration_col, event_col)
            results['proportional_hazards_test'] = ph_test
            
            # Feature importance
            hazard_ratios = self.model.hazard_ratios_
            confidence_intervals = self.model.confidence_intervals_
            
            results['feature_importance'] = {
                'hazard_ratios': hazard_ratios.to_dict(),
                'confidence_intervals': {
                    feature: [ci[0], ci[1]] for feature, ci in 
                    zip(confidence_intervals.index, confidence_intervals.values)
                },
                'p_values': self.model.summary['p'].to_dict()
            }
            
            self.logger.info(f"Cox model fitted with C-index: {results['c_index']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error fitting Cox model: {e}")
            raise
    
    def predict_survival_function(self, X: pd.DataFrame, 
                                 times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Predict survival function for given observations
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        survival_functions = self.model.predict_survival_function(X, times=times)
        return survival_functions
    
    def predict_hazard_ratios(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict hazard ratios (relative risk) for observations
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        partial_hazards = self.model.predict_partial_hazard(X)
        return partial_hazards
    
    def predict_risk_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict risk scores (linear predictor values)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        risk_scores = X @ self.model.params_
        return risk_scores
    
    def stratified_analysis(self, df: pd.DataFrame, duration_col: str, 
                          event_col: str, strata_col: str) -> Dict[str, Any]:
        """
        Perform stratified survival analysis
        """
        
        results = {
            'strata_column': strata_col,
            'strata_results': {},
            'log_rank_test': None
        }
        
        # Get unique strata
        strata_values = df[strata_col].unique()
        
        # Fit separate models for each stratum
        for stratum in strata_values:
            stratum_data = df[df[strata_col] == stratum].copy()
            
            if len(stratum_data) < 10:  # Minimum sample size
                continue
            
            # Fit Cox model for this stratum
            stratum_model = CoxPHFitter(penalizer=0.1)
            
            try:
                feature_cols = [col for col in stratum_data.columns 
                              if col not in [duration_col, event_col, strata_col]]
                
                stratum_model.fit(stratum_data, 
                                duration_col=duration_col, 
                                event_col=event_col)
                
                # Evaluate stratum model
                c_index = concordance_index(
                    stratum_data[duration_col],
                    -stratum_model.predict_partial_hazard(stratum_data),
                    stratum_data[event_col]
                )
                
                results['strata_results'][str(stratum)] = {
                    'c_index': float(c_index),
                    'sample_size': len(stratum_data),
                    'events': int(stratum_data[event_col].sum()),
                    'median_survival': float(stratum_data[duration_col].median()),
                    'hazard_ratios': stratum_model.hazard_ratios_.to_dict()
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to fit model for stratum {stratum}: {e}")
                continue
        
        # Log-rank test between strata
        if len(strata_values) > 1:
            try:
                # Prepare data for log-rank test
                groups = []
                durations = []
                events = []
                
                for stratum in strata_values:
                    stratum_data = df[df[strata_col] == stratum]
                    if len(stratum_data) > 0:
                        groups.extend([stratum] * len(stratum_data))
                        durations.extend(stratum_data[duration_col].tolist())
                        events.extend(stratum_data[event_col].tolist())
                
                # Multivariate log-rank test
                log_rank_result = multivariate_logrank_test(
                    durations, groups, events
                )
                
                results['log_rank_test'] = {
                    'test_statistic': float(log_rank_result.test_statistic),
                    'p_value': float(log_rank_result.p_value),
                    'degrees_of_freedom': int(log_rank_result.degrees_of_freedom),
                    'significant': log_rank_result.p_value < 0.05
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to perform log-rank test: {e}")
        
        return results
    
    def time_dependent_analysis(self, df: pd.DataFrame, duration_col: str,
                              event_col: str, time_col: str) -> Dict[str, Any]:
        """
        Analyze time-dependent covariates
        """
        
        # Check for time-varying effects
        results = {
            'time_column': time_col,
            'time_dependent_effects': {}
        }
        
        # Split data into time periods
        time_periods = np.quantile(df[time_col], [0.25, 0.5, 0.75])
        
        for i, threshold in enumerate(time_periods):
            period_name = f"period_{i+1}_until_{threshold:.1f}"
            
            # Subset data for this time period
            period_data = df[df[time_col] <= threshold].copy()
            
            if len(period_data) < 20:
                continue
            
            # Fit Cox model for this period
            try:
                period_model = CoxPHFitter(penalizer=0.1)
                period_model.fit(period_data, 
                               duration_col=duration_col, 
                               event_col=event_col)
                
                c_index = concordance_index(
                    period_data[duration_col],
                    -period_model.predict_partial_hazard(period_data),
                    period_data[event_col]
                )
                
                results['time_dependent_effects'][period_name] = {
                    'c_index': float(c_index),
                    'sample_size': len(period_data),
                    'time_threshold': float(threshold),
                    'top_risk_factors': period_model.hazard_ratios_.nlargest(5).to_dict()
                }
                
            except Exception as e:
                self.logger.warning(f"Failed analysis for time period {period_name}: {e}")
        
        return results
    
    def _handle_missing_values(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Handle missing values in features"""
        
        df_clean = df.copy()
        
        for col in feature_cols:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'int64']:
                    # Numerical: fill with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # Categorical: fill with mode
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def _engineer_survival_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Engineer features for survival analysis"""
        
        df_eng = df.copy()
        
        # Create interaction terms for key risk factors
        risk_features = [col for col in feature_cols if 'speed' in col.lower() or 
                        'acceleration' in col.lower() or 'brake' in col.lower()]
        
        if len(risk_features) >= 2:
            # Create interaction between top risk features
            df_eng[f'{risk_features[0]}_x_{risk_features[1]}'] = (
                df_eng[risk_features[0]] * df_eng[risk_features[1]]
            )
        
        # Create risk score bins
        for col in feature_cols:
            if df_eng[col].dtype in ['float64', 'int64']:
                # Create quantile-based bins
                try:
                    df_eng[f'{col}_quartile'] = pd.qcut(df_eng[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                    df_eng[f'{col}_quartile'] = df_eng[f'{col}_quartile'].cat.codes
                except:
                    continue
        
        return df_eng
    
    def _evaluate_model(self, df: pd.DataFrame, duration_col: str, event_col: str) -> Dict[str, Any]:
        """Evaluate fitted Cox model"""
        
        # Concordance index
        predicted_risks = self.model.predict_partial_hazard(df)
        c_index = concordance_index(df[duration_col], -predicted_risks, df[event_col])
        
        # Model summary statistics
        aic = self.model.AIC_partial_
        log_likelihood = self.model.log_likelihood_
        
        # Number of events and observations
        n_events = df[event_col].sum()
        n_observations = len(df)
        
        return {
            'c_index': float(c_index),
            'aic': float(aic),
            'log_likelihood': float(log_likelihood),
            'n_events': int(n_events),
            'n_observations': int(n_observations),
            'event_rate': float(n_events / n_observations)
        }
    
    def _test_proportional_hazards(self, df: pd.DataFrame, duration_col: str, 
                                 event_col: str) -> Dict[str, Any]:
        """Test proportional hazards assumption"""
        
        try:
            # Test proportional hazards assumption
            ph_test = self.model.check_assumptions(df, show_plots=False)
            
            return {
                'test_statistic': float(ph_test.test_statistic),
                'p_value': float(ph_test.p_value),
                'assumptions_met': ph_test.p_value > 0.05,
                'summary': ph_test.summary.to_dict() if hasattr(ph_test, 'summary') else {}
            }
            
        except Exception as e:
            self.logger.warning(f"Proportional hazards test failed: {e}")
            return {'test_failed': True, 'error': str(e)}
