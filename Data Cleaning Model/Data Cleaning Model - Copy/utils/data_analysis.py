import numpy as np
import pandas as pd

def analyze_columns(df):
    """Analyze DataFrame columns and return comprehensive statistics."""
    try:
        analysis = {
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_counts': df.nunique().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'column_stats': {}
        }

        # Analyze numeric columns
        for col in analysis['numeric_columns']:
            if not df[col].isnull().all():
                analysis['column_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }

        # Analyze categorical columns
        for col in analysis['categorical_columns']:
            analysis['column_stats'][col] = {
                'top_values': df[col].value_counts().head(5).to_dict()
            }

        return analysis
    except Exception as e:
        print(f"Error in analyze_columns: {str(e)}")
        return None
