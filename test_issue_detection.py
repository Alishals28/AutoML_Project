"""Test STEP 2: Refactored Issue Detection"""
import pandas as pd
from src.utils.feature_type_inference import FeatureTypeInference
from src.components.issue_detection import IssueDetector

# Load heart.csv
df = pd.read_csv('datasets/heart.csv')
print(f"Loaded dataset: {df.shape}")

# Initialize issue detector
detector = IssueDetector(df, target_column='target')

# Detect all issues
issues, suggestions = detector.detect_all_issues()

# Display results
print(f"\nSTEP 2 Test: Issue Detection")
print("="*80)
print(f"Total issues detected: {len(issues)}")
print(f"Total suggestions: {len(suggestions)}")

# Show issue summary
summary = detector.get_issue_summary()
print(f"\nIssue Summary:")
for issue_type, issues_list in summary.items():
    print(f"  - {issue_type}: {len(issues_list)}")

# Verify correct detection
print("\n" + "="*80)
print("VALIDATION CHECKS:")
print("="*80)

# Check 1: Outliers should ONLY be detected on continuous_numeric
outlier_issues = [i for i in issues if i['type'] == 'Outliers']
print(f"\nOutlier Detection ({len(outlier_issues)} issues):")
for issue in outlier_issues:
    col = issue['column']
    ftype = issue['feature_type']
    print(f"  - {col}: {ftype} ({issue['count']} outliers, {issue['percent']}%)")
    
    # Validate: should be continuous_numeric only
    if ftype not in ['continuous_numeric']:
        print(f"    ERROR: Outliers detected on non-continuous column!")
    else:
        print(f"    CORRECT: Outliers on continuous_numeric")

# Check 2: Binary/categorical columns should NOT have outliers
categorical_cols = ['sex', 'cp', 'fbs', 'exang', 'slope', 'ca', 'thal']
print(f"\n✓ Binary/Categorical Outlier Check:")
for col in categorical_cols:
    has_outlier_issue = any(i['column'] == col and i['type'] == 'Outliers' for i in issues)
    if has_outlier_issue:
        print(f"  {col}: HAS outlier issue (WRONG)")
    else:
        print(f"  {col}: No outlier issue (CORRECT)")

# Check 3: Constant/near-constant detection
constant_issues = [i for i in issues if i['type'] in ['Constant Feature', 'Near-Constant Feature']]
print(f"\n✓ Constant Feature Detection ({len(constant_issues)} issues):")
if len(constant_issues) == 0:
    print("  CORRECT: No constant/near-constant features found (as expected for heart.csv)")
else:
    for issue in constant_issues:
        print(f"  - {issue['column']}: {issue['top_frequency_percent']}% ({issue['unique_values']} unique)")

# Check 4: Missing values
missing_issues = [i for i in issues if i['type'] == 'Missing Values']
print(f"\n✓ Missing Values Detection ({len(missing_issues)} issues):")
if len(missing_issues) == 0:
    print("  CORRECT: No missing values (heart.csv is clean)")
else:
    for issue in missing_issues:
        print(f"  - {issue['column']}: {issue['count']} missing ({issue['percent']}%)")

# Check 5: Class imbalance
imbalance_issues = [i for i in issues if i['type'] == 'Class Imbalance']
print(f"\n✓ Class Imbalance Detection ({len(imbalance_issues)} issues):")
if len(imbalance_issues) > 0:
    for issue in imbalance_issues:
        print(f"  Detected: Imbalance ratio {issue['imbalance_ratio']}:1 ({issue['minority_percent']}% minority)")
else:
    print("No significant class imbalance detected")

print("\n" + "="*80)
print("STEP 2 TEST COMPLETE")
print("="*80)
