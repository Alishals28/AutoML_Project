"""
Comprehensive Data Quality Issue Detection

This module detects various data quality issues in a dataset:
1. Missing Values - per feature and global
2. Outliers - IQR method (ONLY on continuous/discrete numeric features)
3. Constant/Near-constant features - dominance-based detection
4. Imbalanced Classes - for classification targets
5. High Cardinality - features with too many unique categories
6. Low Variance - features with very little variation

All detections are feature-type-aware and return structured issue data
with severity levels and actionable suggestions.
"""

import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.feature_type_inference import FeatureTypeInference
from src.utils.outlier_detection import OutlierDetector, ConstantFeatureDetector


class IssueDetector:
    """
    Comprehensive data quality issue detection using feature-type-aware logic.
    
    Returns all detected issues in a structured format with:
    - Issue type and column name
    - Feature type (for context)
    - Severity level
    - Detected metrics (counts, ratios, etc.)
    - Suggested actions with recommendations
    """
    
    def __init__(self, df, target_column=None):
        """
        Initialize issue detector.
        
        Args:
            df (pd.DataFrame): The input dataframe
            target_column (str): Name of target column (excluded from some checks)
        """
        self.df = df
        self.target_column = target_column
        self.issues = []
        self.suggestions = []
        
        logging.info(f"IssueDetector initialized for dataset with {len(df)} rows, {len(df.columns)} columns")
        
        # Initialize feature type inference
        self.feature_type_inference = FeatureTypeInference(df, target_column)
        self.feature_types = self.feature_type_inference.infer_types()
        logging.info(f"Feature types inferred: {len(self.feature_types)} features classified")
        
        # Initialize detectors
        self.outlier_detector = OutlierDetector(df, self.feature_types)
        self.constant_detector = ConstantFeatureDetector(df, self.feature_types)
    
    def detect_all_issues(self):
        """
        Run all issue detection methods in sequence.
        
        Returns:
            tuple: (issues, suggestions)
                   issues = list of dicts with detected issues
                   suggestions = list of dicts with actionable suggestions
        """
        logging.info("Starting comprehensive issue detection...")
        
        self.detect_missing_values()
        self.detect_outliers()  # Now feature-type-aware
        self.detect_constant_features()  # Now dominance-based
        self.detect_imbalanced_classes()
        self.detect_high_cardinality()
        
        logging.info(f"Issue detection complete. Found {len(self.issues)} issues")
        return self.issues, self.suggestions
    
    # ==================== MISSING VALUES ====================
    def detect_missing_values(self):
        """Check 1: Missing Values (per feature + global %)"""
        logging.info("Detecting missing values...")
        
        missing_data = self.df.isnull().sum()
        total_cells = len(self.df)
        global_missing_percent = (missing_data.sum() / (len(self.df) * len(self.df.columns))) * 100
        
        found_count = 0
        for col in missing_data[missing_data > 0].index:
            if col == self.target_column:
                continue
            
            missing_count = missing_data[col]
            missing_percent = (missing_count / total_cells) * 100
            feature_info = self.feature_types.get(col, {})
            feature_type = feature_info.get('type', 'unknown')
            
            # Determine severity
            if missing_percent > 50:
                severity = 'high'
            elif missing_percent > 10:
                severity = 'medium'
            else:
                severity = 'low'
            
            issue = {
                'issue_id': f"missing_{col}",
                'type': 'Missing Values',
                'column': col,
                'feature_type': feature_type,
                'count': int(missing_count),
                'percent': round(missing_percent, 2),
                'severity': severity
            }
            self.issues.append(issue)
            found_count += 1
            
            # Recommendation based on feature type and missing percentage
            if feature_type in ['continuous_numeric', 'discrete_numeric']:
                default_action = 'median'
                options = ['median', 'mean', 'constant value']
            elif feature_type == 'categorical_text':
                default_action = 'mode'
                options = ['mode', 'constant value', 'drop rows']
            else:
                default_action = 'median'
                options = ['median', 'mean', 'mode', 'constant value', 'drop rows']
            
            suggestion = {
                'issue_id': f"missing_{col}",
                'type': 'Missing Values',
                'column': col,
                'options': options,
                'recommended': default_action
            }
            self.suggestions.append(suggestion)
        
        if found_count > 0:
            logging.info(f"Found missing values in {found_count} columns (global: {global_missing_percent:.2f}%)")
        else:
            logging.info("No missing values detected")
    
    # ==================== OUTLIERS (FEATURE-TYPE-AWARE) ====================
    def detect_outliers(self):
        """Check 2: Outliers using IQR (ONLY on continuous/discrete numeric)"""
        logging.info("Detecting outliers (continuous_numeric features only)...")
        
        outlier_results = self.outlier_detector.detect_all_outliers(include_discrete=False)
        
        found_count = 0
        for col, outlier_info in outlier_results.items():
            if not outlier_info.get('has_outliers'):
                continue
            
            feature_info = self.feature_types.get(col, {})
            feature_type = feature_info.get('type', 'unknown')
            
            issue = {
                'issue_id': f"outliers_{col}",
                'type': 'Outliers',
                'column': col,
                'feature_type': feature_type,
                'count': outlier_info.get('outlier_count', 0),
                'percent': outlier_info.get('outlier_percent', 0.0),
                'bounds': {
                    'lower': outlier_info.get('lower_bound'),
                    'upper': outlier_info.get('upper_bound')
                },
                'severity': outlier_info.get('severity', 'low')
            }
            self.issues.append(issue)
            found_count += 1
            
            # Recommendation based on severity
            severity = outlier_info.get('severity', 'low')
            if severity == 'low':
                recommended = 'cap (IQR)'  # Low severity: cap outliers
            elif severity == 'medium':
                recommended = 'cap (IQR)'  # Medium: cap is safer than remove
            else:
                recommended = 'no action'  # High: let user decide
            
            suggestion = {
                'issue_id': f"outliers_{col}",
                'type': 'Outliers',
                'column': col,
                'options': ['remove', 'cap (IQR)', 'no action'],
                'recommended': recommended
            }
            self.suggestions.append(suggestion)
        
        if found_count > 0:
            logging.info(f"Detected outliers in {found_count} columns")
    
    # ==================== IMBALANCED CLASSES ====================
    def detect_imbalanced_classes(self):
        """Check 4: Class imbalance in target variable"""
        if self.target_column is None:
            logging.debug("No target column specified, skipping class imbalance check")
            return
        
        logging.info("Checking for class imbalance...")
        
        if self.target_column not in self.df.columns:
            logging.warning(f"Target column '{self.target_column}' not found")
            return
        
        target_data = self.df[self.target_column].dropna()
        if len(target_data) == 0:
            logging.warning(f"Target column '{self.target_column}' has no valid values")
            return
        
        value_counts = target_data.value_counts()
        min_class_count = value_counts.min()
        max_class_count = value_counts.max()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        # Threshold for imbalance: 3:1 ratio
        if imbalance_ratio > 3.0:
            min_class_percent = (min_class_count / len(target_data)) * 100
            
            severity = 'high' if imbalance_ratio > 10 else 'medium'
            
            issue = {
                'issue_id': 'class_imbalance',
                'type': 'Class Imbalance',
                'column': self.target_column,
                'imbalance_ratio': round(imbalance_ratio, 2),
                'minority_percent': round(min_class_percent, 2),
                'severity': severity
            }
            self.issues.append(issue)
            
            suggestion = {
                'issue_id': 'class_imbalance',
                'type': 'Class Imbalance',
                'column': self.target_column,
                'options': ['class_weights', 'undersampling', 'oversampling', 'no action'],
                'recommended': 'class_weights'
            }
            self.suggestions.append(suggestion)
            
            logging.info(f"Detected class imbalance (ratio: {imbalance_ratio:.2f}:1, severity: {severity})")
        else:
            logging.info(f"Classes are balanced (imbalance ratio: {imbalance_ratio:.2f}:1)")
    
    # ==================== HIGH CARDINALITY ====================
    def detect_high_cardinality(self):
        """Check 5: High cardinality categorical features"""
        logging.info("Detecting high cardinality features...")
        
        found_count = 0
        for col, feature_info in self.feature_types.items():
            feature_type = feature_info.get('type', 'unknown')
            
            # Only check categorical features
            if feature_type not in ['categorical_text', 'categorical_encoded']:
                continue
            
            n_unique = feature_info.get('n_unique', 0)
            n_rows = len(self.df)
            
            # Threshold: >50 unique values OR >20% of rows are unique
            if n_unique > 50 or (n_unique / n_rows > 0.2):
                unique_ratio = round((n_unique / n_rows) * 100, 2)
                
                severity = 'high' if n_unique > 100 else 'medium'
                
                issue = {
                    'issue_id': f"high_cardinality_{col}",
                    'type': 'High Cardinality',
                    'column': col,
                    'feature_type': feature_type,
                    'unique_values': n_unique,
                    'unique_ratio_percent': unique_ratio,
                    'severity': severity
                }
                self.issues.append(issue)
                found_count += 1
                
                suggestion = {
                    'issue_id': f"high_cardinality_{col}",
                    'type': 'High Cardinality',
                    'column': col,
                    'options': ['group_rare_categories', 'drop_feature', 'keep_as_is'],
                    'recommended': 'group_rare_categories'
                }
                self.suggestions.append(suggestion)
        
        if found_count > 0:
            logging.info(f"Detected high cardinality in {found_count} features")
        else:
            logging.info("No high cardinality features detected")
    
    # ==================== CONSTANT/NEAR-CONSTANT FEATURES ====================
    def detect_constant_features(self):
        """Check 3: Constant/Near-constant features using dominance logic"""
        logging.info("Detecting constant/near-constant features...")
        
        constant_results = self.constant_detector.detect_all_constant_features()
        
        for col, const_info in constant_results.items():
            feature_type = const_info.get('feature_type', 'unknown')
            top_freq = const_info.get('top_frequency_percent', 0)
            
            if const_info.get('is_constant'):
                severity = 'high'
                issue_type = 'Constant Feature'
            else:
                severity = 'medium'
                issue_type = 'Near-Constant Feature'
            
            issue = {
                'issue_id': f"constant_{col}",
                'type': issue_type,
                'column': col,
                'feature_type': feature_type,
                'top_frequency_percent': round(top_freq, 2),
                'top_value': str(const_info.get('top_value', 'N/A')),
                'unique_values': const_info.get('n_unique', 1),
                'severity': severity
            }
            self.issues.append(issue)
            
            suggestion = {
                'issue_id': f"constant_{col}",
                'type': issue_type,
                'column': col,
                'options': ['drop feature', 'keep feature'],
                'recommended': 'drop feature'  # Default: drop constant features
            }
            self.suggestions.append(suggestion)
        
        if len(constant_results) > 0:
            logging.info(f"Detected {len(constant_results)} constant/near-constant features")
        else:
            logging.info("No constant/near-constant features detected")
    
    # ==================== UTILITY METHODS ====================
    def get_issue_summary(self):
        """Get a summary of all detected issues grouped by type"""
        summary = {}
        for issue in self.issues:
            issue_type = issue.get('type', 'Unknown')
            if issue_type not in summary:
                summary[issue_type] = []
            summary[issue_type].append(issue)
        
        return summary
    
    def get_suggestion_for_issue(self, issue_id):
        """Get the suggestion for a specific issue"""
        for sug in self.suggestions:
            if sug.get('issue_id') == issue_id:
                return sug
        return None
    
    def get_issues_by_column(self, col):
        """Get all issues detected for a specific column"""
        return [issue for issue in self.issues if issue.get('column') == col]
    
    def display_issues(self):
        """Print a human-readable summary of all detected issues"""
        if not self.issues:
            print("\n✅ No data quality issues detected!\n")
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY ISSUES DETECTED")
        print("="*80)
        
        summary = self.get_issue_summary()
        for issue_type, issues_list in summary.items():
            print(f"\n🔴 {issue_type.upper()} ({len(issues_list)} issues)")
            print("-" * 80)
            
            for issue in issues_list:
                col = issue.get('column', 'N/A')
                severity = issue.get('severity', 'unknown').upper()
                feature_type = issue.get('feature_type', 'unknown')
                
                print(f"  Column: {col}")
                print(f"  Feature Type: {feature_type}")
                print(f"  Severity: {severity}")
                
                if issue_type == 'Missing Values':
                    print(f"  Missing: {issue.get('count')} ({issue.get('percent')}%)")
                elif issue_type == 'Outliers':
                    print(f"  Outliers: {issue.get('count')} ({issue.get('percent')}%)")
                    print(f"  Bounds: [{issue.get('bounds', {}).get('lower')}, {issue.get('bounds', {}).get('upper')}]")
                elif issue_type in ['Constant Feature', 'Near-Constant Feature']:
                    print(f"  Top Value: {issue.get('top_value')} ({issue.get('top_frequency_percent')}%)")
                    print(f"  Unique Values: {issue.get('unique_values')}")
                elif issue_type == 'Class Imbalance':
                    print(f"  Imbalance Ratio: {issue.get('imbalance_ratio')}:1")
                    print(f"  Minority Class: {issue.get('minority_percent')}%")
                elif issue_type == 'High Cardinality':
                    print(f"  Unique Values: {issue.get('unique_values')} ({issue.get('unique_ratio_percent')}%)")
                
                print()
        
        print("="*80 + "\n")