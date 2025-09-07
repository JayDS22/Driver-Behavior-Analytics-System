import numpy as np
import pandas as pd
from lifelines import WeibullFitter, LogNormalFitter, ExponentialFitter
from lifelines.utils import concordance_index
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import logging

class ParametricSurvivalModels:
    """
    Parametric survival models for driver risk analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fitted_models = {}
        
    def fit_weibull_model(self, durations: np.ndarray, events: np.ndarray,
                         label: str = "Weibull") -> Dict[str, Any]:
        """
        Fit Weibull survival model
        """
        
        wf = WeibullFitter()
        wf.fit(durations, events, label=label)
        
        self.fitted_models[label] = wf
        
        # Model parameters
        lambda_param = wf.lambda_
        rho_param = wf.rho_
        
        # Model evaluation
        aic = wf.AIC_
        
        # Median survival time
        median_survival = wf.median_survival_time_
        
        # Survival function at specific timepoints
        timepoints = self.config.get('survival_timepoints', [30, 60, 90, 180, 365])
        survival_at_timepoints = {}
        
        for t in timepoints:
            survival_prob = wf.survival_function_at_times(t).iloc[0]
            survival_at_timepoints[f'survival_at_{t}'] = float(survival_prob)
        
        # Hazard function values
        hazard_at_timepoints = {}
        for t in timepoints:
            hazard_rate = wf.hazard_at_times(t).iloc[0]
            hazard_at_timepoints[f'hazard_at_{t}'] = float(hazard_rate)
        
        return {
            'model_type': 'Weibull',
            'label': label,
            'parameters': {
                'lambda': float(lambda_param),
                'rho': float(rho_param)
            },
            'aic': float(aic),
            'median_survival_time': float(median_survival) if not pd.isna(median_survival) else None,
            'survival_at_timepoints': survival_at_timepoints,
            'hazard_at_timepoints': hazard_at_timepoints,
            'n_observations': len(durations),
            'n_events': int(events.sum())
        }
    
    def fit_lognormal_model(self, durations: np.ndarray, events: np.ndarray,
                           label: str = "LogNormal") -> Dict[str, Any]:
        """
        Fit Log-Normal survival model
        """
        
        lnf = LogNormalFitter()
        lnf.fit(durations, events, label=label)
        
        self.fitted_models[label] = lnf
        
        # Model parameters
        mu_param = lnf.mu_
        sigma_param = lnf.sigma_
        
        # Model evaluation
        aic = lnf.AIC_
        median_survival = lnf.median_survival_time_
        
        # Survival and hazard at timepoints
        timepoints = self.config.get('survival_timepoints', [30, 60, 90, 180, 365])
        survival_at_timepoints = {}
        hazard_at_timepoints = {}
        
        for t in timepoints:
            survival_prob = lnf.survival_function_at_times(t).iloc[0]
            hazard_rate = lnf.hazard_at_times(t).iloc[0]
            
            survival_at_timepoints[f'survival_at_{t}'] = float(survival_prob)
            hazard_at_timepoints[f'hazard_at_{t}'] = float(hazard_rate)
        
        return {
            'model_type': 'LogNormal',
            'label': label,
            'parameters': {
                'mu': float(mu_param),
                'sigma': float(sigma_param)
            },
            'aic': float(aic),
            'median_survival_time': float(median_survival) if not pd.isna(median_survival) else None,
            'survival_at_timepoints': survival_at_timepoints,
            'hazard_at_timepoints': hazard_at_timepoints,
            'n_observations': len(durations),
            'n_events': int(events.sum())
        }
    
    def fit_exponential_model(self, durations: np.ndarray, events: np.ndarray,
                             label: str = "Exponential") -> Dict[str, Any]:
        """
        Fit Exponential survival model
        """
        
        ef = ExponentialFitter()
        ef.fit(durations, events, label=label)
        
        self.fitted_models[label] = ef
        
        # Model parameters
        lambda_param = ef.lambda_
        
        # Model evaluation
        aic = ef.AIC_
        median_survival = ef.median_survival_time_
        
        # For exponential model, hazard is constant = lambda
        constant_hazard = float(lambda_param)
        
        # Survival at timepoints
        timepoints = self.config.get('survival_timepoints', [30, 60, 90, 180, 365])
        survival_at_timepoints = {}
        
        for t in timepoints:
            survival_prob = ef.survival_function_at_times(t).iloc[0]
            survival_at_timepoints[f'survival_at_{t}'] = float(survival_prob)
        
        return {
            'model_type': 'Exponential',
            'label': label,
            'parameters': {
                'lambda': float(lambda_param)
            },
            'aic': float(aic),
            'median_survival_time': float(median_survival) if not pd.isna(median_survival) else None,
            'constant_hazard_rate': constant_hazard,
            'survival_at_timepoints': survival_at_timepoints,
            'n_observations': len(durations),
            'n_events': int(events.sum())
        }
    
    def compare_models(self, durations: np.ndarray, events: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple parametric survival models
        """
        
        models_to_fit = ['Weibull', 'LogNormal', 'Exponential']
        model_results = {}
        
        # Fit all models
        for model_type in models_to_fit:
            try:
                if model_type == 'Weibull':
                    result = self.fit_weibull_model(durations, events, model_type)
                elif model_type == 'LogNormal':
                    result = self.fit_lognormal_model(durations, events, model_type)
                elif model_type == 'Exponential':
                    result = self.fit_exponential_model(durations, events, model_type)
                
                model_results[model_type] = result
                
            except Exception as e:
                self.logger.warning(f"Failed to fit {model_type} model: {e}")
                continue
        
        # Model comparison
        if len(model_results) > 1:
            comparison = self._perform_model_comparison(model_results)
        else:
            comparison = {}
        
        return {
            'model_results': model_results,
            'model_comparison': comparison,
            'best_model': comparison.get('best_model_by_aic', 'Unknown'),
            'summary': self._generate_comparison_summary(model_results, comparison)
        }
    
    def _perform_model_comparison(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare models using AIC and other criteria
        """
        
        aic_values = {}
        median_survivals = {}
        
        for model_name, result in model_results.items():
            aic_values[model_name] = result['aic']
            median_survivals[model_name] = result['median_survival_time']
        
        # Best model by AIC (lowest AIC)
        best_model_by_aic = min(aic_values, key=aic_values.get)
        
        # AIC differences
        best_aic = aic_values[best_model_by_aic]
        aic_differences = {
            model: aic - best_aic for model, aic in aic_values.items()
        }
        
        # Model weights using AIC
        aic_weights = {}
        total_weight = sum(np.exp(-0.5 * diff) for diff in aic_differences.values())
        
        for model, diff in aic_differences.items():
            aic_weights[model] = np.exp(-0.5 * diff) / total_weight
        
        return {
            'aic_values': aic_values,
            'best_model_by_aic': best_model_by_aic,
            'aic_differences': aic_differences,
            'aic_weights': aic_weights,
            'median_survival_comparison': median_survivals
        }
    
    def _generate_comparison_summary(self, model_results: Dict[str, Any], 
                                   comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of model comparison
        """
        
        summary = {
            'models_fitted': list(model_results.keys()),
            'total_observations': model_results[list(model_results.keys())[0]]['n_observations'],
            'total_events': model_results[list(model_results.keys())[0]]['n_events']
        }
        
        if comparison:
            summary.update({
                'recommended_model': comparison['best_model_by_aic'],
                'model_confidence': comparison['aic_weights'].get(comparison['best_model_by_aic'], 0),
                'aic_improvement': min(comparison['aic_differences'].values())
            })
        
        return summary
