"""
Outlier and Constant Feature Detection Module

This module provides feature-type-aware detection of:
1. Outliers - using IQR method (ONLY on continuous_numeric features)
2. Constant/Near-constant features - using dominance logic
"""

import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException


class OutlierDetector:
    """
    Detects outliers using the IQR (Interquartile Range) method.
    
    IMPORTANT: Only applies to continuous_numeric and optionally discrete_numeric features.
    Never applies to binary, categorical_encoded, or categorical_text features.
    """
    
    def __init__(self, df, feature_types_dict):
        """
        Initialize outlier detector.
        
        Args:
            df (pd.DataFrame): The input dataframe
            feature_types_dict (dict): Output from FeatureTypeInference.infer_types()
                                       Maps column names to feature type info dicts
        """
        self.df = df
        self.feature_types = feature_types_dict
        logging.info(f"OutlierDetector initialized with {len(df)} rows")
    
    def detect_outliers(self, col, method='iqr'):
        """
        Detect outliers in a column using IQR method.
        
        Args:
            col (str): Column name to check for outliers
            method (str): Detection method (currently only 'iqr' is supported)
        
        Returns:
            dict with outlier statistics, or None if not applicable to this column type
            
            Example return:
            {
                'has_outliers': bool,
                'outlier_count': int,
                'outlier_ratio': float,
                'outlier_percent': float,
                'lower_bound': float,
                'upper_bound': float,
                'Q1': float,
                'Q3': float,
                'IQR': float,
                'severity': 'low' | 'medium' | 'high',
                'outlier_indices': list,
                'min_value': float,
                'max_value': float
            }
        """
        feature_info = self.feature_types.get(col, {})
        feature_type = feature_info.get('type')
        
        # === ONLY detect outliers for continuous/discrete numeric ===
        if feature_type not in ['continuous_numeric', 'discrete_numeric']:
            logging.debug(f"Skipping outlier detection for '{col}' (type: {feature_type})")
            return None  # Not applicable
        
        data = self.df[col].dropna()
        if len(data) == 0:
            logging.warning(f"Column '{col}' has no non-null values")
            return None
        
        # === IQR Method ===
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Handle case where IQR is 0 (no spread - all values same)
        if IQR == 0:
            logging.debug(f"Column '{col}' has zero IQR (no spread in data)")
            return {
                'has_outliers': False,
                'outlier_count': 0,
                'outlier_ratio': 0.0,
                'outlier_percent': 0.0,
                'lower_bound': float(Q1),
                'upper_bound': float(Q3),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': 0.0,
                'severity': 'low',
                'outlier_indices': [],
                'min_value': float(data.min()),
                'max_value': float(data.max()),
                'note': 'No spread in data (IQR=0)'
            }
        
        # Calculate bounds using standard IQR method
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_count = outlier_mask.sum()
        outlier_ratio = outlier_count / len(data)
        
        # Determine severity based on outlier ratio
        if outlier_ratio < 0.01:
            severity = 'low'      # <1% outliers
        elif outlier_ratio < 0.05:
            severity = 'medium'   # 1-5% outliers
        else:
            severity = 'high'     # >5% outliers
        
        result = {
            'has_outliers': outlier_count > 0,
            'outlier_count': int(outlier_count),
            'outlier_ratio': round(outlier_ratio, 4),
            'outlier_percent': round(outlier_ratio * 100, 2),
            'lower_bound': round(lower_bound, 4),
            'upper_bound': round(upper_bound, 4),
            'Q1': round(Q1, 4),
            'Q3': round(Q3, 4),
            'IQR': round(IQR, 4),
            'severity': severity,
            'outlier_indices': outlier_indices,
            'min_value': round(data.min(), 4),
            'max_value': round(data.max(), 4)
        }
        
        logging.info(f"Outlier detection for '{col}': {outlier_count} outliers ({outlier_ratio*100:.2f}%), severity={severity}")
        return result
    
    def detect_all_outliers(self, include_discrete=False):
        """
        Detect outliers in all applicable columns.
        
        Args:
            include_discrete (bool): If True, also check discrete_numeric columns
        
        Returns:
            dict: {column_name: outlier_detection_result} for columns with outliers
                  (only includes columns that actually have outliers)
        """
        results = {}
        applicable_types = ['continuous_numeric']
        if include_discrete:
            applicable_types.append('discrete_numeric')
        
        for col, feature_info in self.feature_types.items():
            if feature_info.get('type') in applicable_types:
                outlier_result = self.detect_outliers(col)
                if outlier_result and outlier_result.get('has_outliers'):
                    results[col] = outlier_result
        
        logging.info(f"Detected outliers in {len(results)} columns")
        return results


class ConstantFeatureDetector:
    """
    Detects constant or near-constant features using dominance logic.
    
    A feature is constant/near-constant if the most common value dominates the dataset.
    This is more robust than unique ratio checks, especially for categorical data.
    """
    
    def __init__(self, df, feature_types_dict, near_constant_threshold=0.99):
        """
        Initialize constant feature detector.
        
        Args:
            df (pd.DataFrame): The input dataframe
            feature_types_dict (dict): Output from FeatureTypeInference.infer_types()
            near_constant_threshold (float): Threshold for near-constant detection
                                             Default 0.99 means ≥99% of values are the same
        """
        self.df = df
        self.feature_types = feature_types_dict
        self.near_constant_threshold = near_constant_threshold
        logging.info(f"ConstantFeatureDetector initialized (threshold={near_constant_threshold})")
    
    def detect_constant_feature(self, col):
        """
        Detect if a column is constant or near-constant using dominance logic.
        
        Args:
            col (str): Column name to check
        
        Returns:
            dict with constant detection info, or None if not constant/near-constant
            
            Example return:
            {
                'is_constant': bool,
                'is_near_constant': bool,
                'top_frequency': float (0.0 to 1.0),
                'top_frequency_percent': float,
                'top_value': any,
                'top_count': int,
                'total_count': int,
                'n_unique': int,
                'severity': 'high' | 'medium' | None,
                'feature_type': str
            }
        """
        feature_info = self.feature_types.get(col, {})
        feature_type = feature_info.get('type', 'unknown')
        
        data = self.df[col].dropna()
        if len(data) == 0:
            logging.warning(f"Column '{col}' has no non-null values")
            return None
        
        # Calculate dominance of most common value
        value_counts = data.value_counts()
        top_count = value_counts.iloc[0]
        top_value = value_counts.index[0]
        top_frequency = top_count / len(data)
        
        is_constant = (top_frequency == 1.0)  # 100% same value
        is_near_constant = (top_frequency >= self.near_constant_threshold)  # ≥99%
        
        # Only return if it's actually constant or near-constant
        if is_constant or is_near_constant:
            severity = 'high' if is_constant else 'medium'
            
            result = {
                'is_constant': is_constant,
                'is_near_constant': is_near_constant,
                'top_frequency': round(top_frequency, 4),
                'top_frequency_percent': round(top_frequency * 100, 2),
                'top_value': top_value,
                'top_count': int(top_count),
                'total_count': len(data),
                'n_unique': data.nunique(),
                'severity': severity,
                'feature_type': feature_type
            }
            
            logging.info(f"Detected constant/near-constant feature '{col}': "
                        f"top_frequency={top_frequency*100:.2f}%, severity={severity}")
            return result
        
        logging.debug(f"Column '{col}' is not constant (top_frequency={top_frequency*100:.2f}%)")
        return None
    
    def detect_all_constant_features(self):
        """
        Detect all constant/near-constant features in the dataframe.
        
        Returns:
            dict: {column_name: constant_detection_result} for constant/near-constant features
        """
        results = {}
        for col in self.feature_types.keys():
            result = self.detect_constant_feature(col)
            if result:
                results[col] = result
        
        logging.info(f"Detected {len(results)} constant/near-constant features")
        return results
