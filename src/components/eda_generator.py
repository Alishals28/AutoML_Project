import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


class EDAGenerator:
    """Comprehensive EDA generator with enhanced visualizations and metrics exposure."""

    def __init__(self, df, target_column=None, test_size: float = 0.2):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.global_missing_percent = self._compute_global_missing_percent()

    def _compute_global_missing_percent(self) -> float:
        """Compute global missing value percentage across entire dataset."""
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        return round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    def get_global_missing_percent(self) -> float:
        """Expose global missing value percentage to UI."""
        return self.global_missing_percent
    
    def generate_basic_stats(self):
        """A. Basic Dataset Info"""
        stats_dict = {
            'Total Rows': len(self.df),
            'Total Columns': len(self.df.columns),
            'Numeric Columns': len(self.numeric_cols),
            'Categorical Columns': len(self.categorical_cols),
            'Memory Usage (MB)': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            'Global Missing %': self.global_missing_percent,
            'Column Types': self.df.dtypes.to_dict()
        }
        return stats_dict
    
    def generate_summary_statistics(self):
        """Summary statistics for numeric columns"""
        return self.df[self.numeric_cols].describe().round(3)
    
    def generate_class_distribution(self):
        """B. Class Distribution with percentages and counts."""
        if self.target_column and self.target_column in self.df.columns:
            class_counts = self.df[self.target_column].value_counts()
            class_percent = (class_counts / len(self.df)) * 100

            # Create a more informative class distribution figure
            class_labels = [f"{cls}<br>({class_percent[cls]:.1f}%)" for cls in class_counts.index]
            fig = go.Figure(data=[
                go.Bar(
                    x=class_labels,
                    y=class_counts.values,
                    text=class_counts.values,
                    textposition='outside',
                    name='Class Count',
                    marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(class_counts)])
                )
            ])
            fig.update_layout(
                title=f'Target Class Distribution ({self.target_column})',
                xaxis_title='Class',
                yaxis_title='Count',
                showlegend=True,
                hovermode='x unified'
            )
            return fig, {
                'counts': class_counts.to_dict(),
                'percentages': class_percent.to_dict(),
                'total_samples': len(self.df),
            }
        return None, None
    
    def generate_missing_value_analysis(self):
        """B.1: Missing Value Analysis (per feature + global % exposure)."""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percent.values
        }).sort_values('Missing %', ascending=False)
        
        global_missing = self.global_missing_percent
        
        if missing_data.sum() > 0:
            fig = px.bar(missing_df[missing_df['Missing %'] > 0], x='Column', y='Missing %',
                        title='Missing Values Analysis',
                        labels={'Missing %': 'Missing Percentage (%)'})
            fig.update_layout(hovermode='x unified')
        else:
            fig = None
        
        analysis_summary = {
            'per_column': missing_df.to_dict('records'),
            'global_missing_percent': global_missing,
            'total_missing_cells': self.df.isnull().sum().sum(),
        }
        
        return fig, analysis_summary
    
    def generate_outlier_analysis(self):
        """B.2: Outlier Detection (IQR and Z-score)"""
        outlier_data = {}
        
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
            
            # IQR Method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            iqr_outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)][col].count()
            
            # Z-score Method
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            zscore_outliers = (z_scores > 3).sum()
            
            outlier_data[col] = {
                'IQR Outliers': iqr_outliers,
                'Z-Score Outliers': zscore_outliers,
                'Lower Bound': round(lower, 3),
                'Upper Bound': round(upper, 3)
            }
        
        # Box plots
        figs = []
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
            fig = px.box(self.df, y=col, title=f'Box Plot: {col}')
            figs.append(fig)
        
        return outlier_data, figs
    
    def generate_correlation_matrix(self):
        """B.3: Correlation Matrix"""
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           labels=dict(color="Correlation"),
                           title='Correlation Matrix',
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
            return fig, corr_matrix
        return None, None
    
    def generate_distribution_plots(self):
        """B.4: Distribution Plots for Numerical Features"""
        figs = []
        
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
            
            fig = make_subplots(specs=[[{'secondary_y': True}]])
            
            fig.add_trace(go.Histogram(x=self.df[col], name='Histogram',
                                      nbinsx=30, opacity=0.7),
                         secondary_y=False)
            fig.add_trace(go.Scatter(x=self.df[col].sort_values(),
                                    y=stats.norm.pdf(self.df[col].sort_values(),
                                                    self.df[col].mean(),
                                                    self.df[col].std()),
                                    name='Normal Distribution', mode='lines'),
                         secondary_y=True)
            
            fig.update_layout(title=f'Distribution: {col}', hovermode='x unified')
            figs.append(fig)
        
        return figs
    
    def generate_categorical_plots(self):
        """B.5: Bar Plots for Categorical Features"""
        figs = []
        
        for col in self.categorical_cols:
            if col == self.target_column:
                continue
            
            value_counts = self.df[col].value_counts().head(10)  # Top 10
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Top 10 Categories: {col}',
                        labels={'x': col, 'y': 'Count'})
            figs.append(fig)
        
        return figs
    
    def generate_train_test_split_summary(self):
        """B.6: Train/Test Split Summary using configured test_size."""
        train_size = int(len(self.df) * (1 - self.test_size))
        test_size_count = len(self.df) - train_size
        
        split_summary = {
            'Total Samples': len(self.df),
            'Train Samples': train_size,
            'Test Samples': test_size_count,
            'Train Ratio %': round((train_size / len(self.df)) * 100, 2),
            'Test Ratio %': round((test_size_count / len(self.df)) * 100, 2),
            'Test Size Parameter': self.test_size,
        }
        
        fig = go.Figure(data=[
            go.Pie(labels=['Train', 'Test'],
                  values=[train_size, test_size_count],
                  textposition='inside',
                  textinfo='label+percent')
        ])
        fig.update_layout(title='Train/Test Split Distribution')
        
        return split_summary, fig