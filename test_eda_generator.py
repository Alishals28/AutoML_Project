"""Test STEP 8: EDA Generator"""
import pandas as pd

from src.components.eda_generator import EDAGenerator


print("Loading datasets/heart.csv ...")
df = pd.read_csv("datasets/heart.csv")
print(f"Dataset shape: {df.shape}")

# Initialize EDA generator with test_size parameter
eda = EDAGenerator(df, target_column="target", test_size=0.25)

print("\n" + "="*80)
print("STEP 8 Test: EDA Generator Enhancements")
print("="*80)

# Test 1: Global missing percent exposure
print("\n1. Global Missing Percent Exposure:")
global_missing = eda.get_global_missing_percent()
print(f"   Global missing percent: {global_missing}%")
assert isinstance(global_missing, float), "Global missing % must be float"

# Test 2: Basic stats includes global missing %
print("\n2. Basic Stats with Global Missing %:")
basic_stats = eda.generate_basic_stats()
print(f"   Total rows: {basic_stats['Total Rows']}")
print(f"   Total columns: {basic_stats['Total Columns']}")
print(f"   Global Missing % in stats: {basic_stats['Global Missing %']}")
assert "Global Missing %" in basic_stats, "Global missing % not in basic stats"

# Test 3: Class distribution with percentages
print("\n3. Class Distribution (enhanced):")
class_fig, class_info = eda.generate_class_distribution()
if class_fig:
    print(f"   Class counts: {class_info['counts']}")
    print(f"   Class percentages: {class_info['percentages']}")
    print(f"   Total samples: {class_info['total_samples']}")
    assert "total_samples" in class_info, "Missing total_samples in class_info"
    assert "percentages" in class_info, "Missing percentages in class_info"

# Test 4: Missing value analysis with global exposure
print("\n4. Missing Value Analysis:")
missing_fig, missing_summary = eda.generate_missing_value_analysis()
print(f"   Global missing percent: {missing_summary['global_missing_percent']}%")
print(f"   Total missing cells: {missing_summary['total_missing_cells']}")
assert "global_missing_percent" in missing_summary, "Missing global_missing_percent"

# Test 5: Train/test split summary with test_size parameter
print("\n5. Train/Test Split Summary (with test_size=0.25):")
split_summary, split_fig = eda.generate_train_test_split_summary()
print(f"   Total samples: {split_summary['Total Samples']}")
print(f"   Train samples: {split_summary['Train Samples']}")
print(f"   Test samples: {split_summary['Test Samples']}")
print(f"   Train ratio %: {split_summary['Train Ratio %']}%")
print(f"   Test ratio %: {split_summary['Test Ratio %']}%")
print(f"   Test size parameter used: {split_summary['Test Size Parameter']}")
assert split_summary['Test Size Parameter'] == 0.25, "Test size parameter not aligned"
assert abs(split_summary['Test Ratio %'] - 25.0) < 1, "Test ratio should be ~25%"

# Test 6: Other visualizations remain available
print("\n6. Other Visualizations:")
summary_stats = eda.generate_summary_statistics()
print(f"   Summary stats shape: {summary_stats.shape}")

correlation_fig, corr_matrix = eda.generate_correlation_matrix()
print(f"   Correlation matrix computed: {corr_matrix is not None}")

outlier_data, outlier_figs = eda.generate_outlier_analysis()
print(f"   Outlier analysis completed: {len(outlier_data)} numeric columns analyzed")

distribution_figs = eda.generate_distribution_plots()
print(f"   Distribution plots generated: {len(distribution_figs)} plots")

categorical_figs = eda.generate_categorical_plots()
print(f"   Categorical plots generated: {len(categorical_figs)} plots")

print("\n" + "="*80)
print("STEP 8 TEST PASSED: EDA generator with enhanced class distribution, global missing %, and test_size.")
print("="*80)
