import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, pairwise_logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging

class KaplanMeierAnalysis:
    """
    Kaplan-Meier survival analysis for driver behavior
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fitted_models = {}
    
    def fit_survival_function(self, durations: np.ndarray, events: np.ndarray,
                            label: str = "Population") -> Dict[str, Any]:
        """
        Fit Kaplan-Meier survival function
        
        Args:
            durations: Time to event or censoring
            events: Event indicator (1=event, 0=censored)
            label: Label for this survival function
            
        Returns:
            Dictionary with survival function and statistics
        """
        
        kmf = KaplanMeierFitter()
        kmf.fit(durations, events, label=label)
        
        # Store fitted model
        self.fitted_models[label] = kmf
        
        # Calculate statistics
        median_survival = kmf.median_survival_time_
        survival_at_timepoints = self._calculate_survival_at_timepoints(kmf)
        
        # Confidence intervals
        confidence_interval = kmf.confidence_interval_
        
        results = {
            'label': label,
            'n_observations': len(durations),
            'n_events': int(events.sum()),
            'median_survival_time': float(median_survival) if not pd.isna(median_survival) else None,
            'survival_function': {
                'timeline': kmf.timeline.tolist(),
                'survival_prob': kmf.survival_function_[label].tolist(),
                'confidence_interval_lower': confidence_interval[f'{label}_lower_0.95'].tolist(),
                'confidence_interval_upper': confidence_interval[f'{label}_upper_0.95'].tolist()
            },
            'survival_at_timepoints': survival_at_timepoints,
            'event_rate': float(events.sum() / len(events))
        }
        
        self.logger.info(f"Fitted KM survival function for {label}: "
                        f"median survival = {median_survival}")
        
        return results
    
    def compare_groups(self, df: pd.DataFrame, duration_col: str, 
                      event_col: str, group_col: str) -> Dict[str, Any]:
        """
        Compare survival functions between groups
        """
        
        groups = df[group_col].unique()
        group_results = {}
        
        # Fit survival function for each group
        for group in groups:
            group_data = df[df[group_col] == group]
            durations = group_data[duration_col].values
            events = group_data[event_col].values
            
            if len(durations) > 0:
                group_results[str(group)] = self.fit_survival_function(
                    durations, events, label=str(group)
                )
        
        # Perform log-rank tests
        log_rank_results = self._perform_log_rank_tests(df, duration_col, event_col, group_col)
        
        # Calculate hazard ratios between groups
        hazard_ratios = self._calculate_hazard_ratios(group_results)
        
        comparison_results = {
            'group_column': group_col,
            'group_results': group_results,
            'log_rank_tests': log_rank_results,
            'hazard_ratios': hazard_ratios,
            'summary_statistics': self._calculate_group_summary(group_results)
        }
        
        return comparison_results
    
    def stratified_analysis(self, df: pd.DataFrame, duration_col: str,
                          event_col: str, strata_cols: List[str]) -> Dict[str, Any]:
        """
        Perform stratified Kaplan-Meier analysis
        """
        
        results = {
            'strata_columns': strata_cols,
            'strata_results': {}
        }
        
        # Create strata combinations
        if len(strata_cols) == 1:
            strata_values = df[strata_cols[0]].unique()
            strata_combinations = [(col, val) for val in strata_values for col in [strata_cols[0]]]
        else:
            # Multiple strata - create combinations
            strata_combinations = []
            for _, row in df[strata_cols].drop_duplicates().iterrows():
                strata_combinations.append(tuple(row.values))
        
        # Analyze each stratum
        for stratum in strata_combinations:
            if len(strata_cols) == 1:
                stratum_data = df[df[strata_cols[0]] == stratum[1]]
                stratum_label = f"{strata_cols[0]}_{stratum[1]}"
            else:
                mask = pd.Series([True] * len(df))
                for i, col in enumerate(strata_cols):
                    mask &= (df[col] == stratum[i])
                stratum_data = df[mask]
                stratum_label = "_".join([f"{col}_{val}" for col, val in zip(strata_cols, stratum)])
            
            if len(stratum_data) >= 10:  # Minimum sample size
                durations = stratum_data[duration_col].values
                events = stratum_data[event_col].values
                
                stratum_result = self.fit_survival_function(durations, events, stratum_label)
                results['strata_results'][stratum_label] = stratum_result
        
        return results
    
    def _calculate_survival_at_timepoints(self, kmf: KaplanMeierFitter) -> Dict[str, float]:
        """Calculate survival probabilities at specific timepoints"""
        
        timepoints = self.config.get('survival_timepoints', [30, 60, 90, 180, 365])
        survival_at_timepoints = {}
        
        for timepoint in timepoints:
            try:
                survival_prob = kmf.survival_function_at_times(timepoint).iloc[0]
                survival_at_timepoints[f'survival_at_{timepoint}_days'] = float(survival_prob)
            except:
                survival_at_timepoints[f'survival_at_{timepoint}_days'] = None
        
        return survival_at_timepoints
    
    def _perform_log_rank_tests(self, df: pd.DataFrame, duration_col: str,
                               event_col: str, group_col: str) -> Dict[str, Any]:
        """Perform log-rank tests between groups"""
        
        groups = df[group_col].unique()
        log_rank_results = {}
        
        # Pairwise log-rank tests
        if len(groups) > 2:
            try:
                pairwise_results = pairwise_logrank_test(
                    df[duration_col], df[group_col], df[event_col]
                )
                
                log_rank_results['pairwise_tests'] = {
                    'test_statistics': pairwise_results.test_statistic.to_dict(),
                    'p_values': pairwise_results.p_value.to_dict()
                }
            except Exception as e:
                self.logger.warning(f"Pairwise log-rank test failed: {e}")
        
        # Overall log-rank test
        if len(groups) == 2:
            group1_data = df[df[group_col] == groups[0]]
            group2_data = df[df[group_col] == groups[1]]
            
            try:
                log_rank_result = logrank_test(
                    group1_data[duration_col], group2_data[duration_col],
                    group1_data[event_col], group2_data[event_col]
                )
                
                log_rank_results['overall_test'] = {
                    'test_statistic': float(log_rank_result.test_statistic),
                    'p_value': float(log_rank_result.p_value),
                    'significant': log_rank_result.p_value < 0.05
                }
            except Exception as e:
                self.logger.warning(f"Overall log-rank test failed: {e}")
        
        return log_rank_results
    
    def _calculate_hazard_ratios(self, group_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate approximate hazard ratios between groups"""
        
        hazard_ratios = {}
        
        # Use median survival times to approximate hazard ratios
        medians = {}
        for group, result in group_results.items():
            median_survival = result.get('median_survival_time')
            if median_survival is not None:
                medians[group] = median_survival
        
        # Calculate ratios relative to first group
        if len(medians) > 1:
            reference_group = list(medians.keys())[0]
            reference_median = medians[reference_group]
            
            for group, median in medians.items():
                if group != reference_group and median > 0:
                    # Approximate hazard ratio
                    hazard_ratio = reference_median / median
                    hazard_ratios[f'{group}_vs_{reference_group}'] = float(hazard_ratio)
        
        return hazard_ratios
    
    def _calculate_group_summary(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across groups"""
        
        summary = {
            'total_observations': sum(result['n_observations'] for result in group_results.values()),
            'total_events': sum(result['n_events'] for result in group_results.values()),
            'group_statistics': {}
        }
        
        for group, result in group_results.items():
            summary['group_statistics'][group] = {
                'sample_size': result['n_observations'],
                'events': result['n_events'],
                'event_rate': result['event_rate'],
                'median_survival': result['median_survival_time']
            }
        
        return summary
