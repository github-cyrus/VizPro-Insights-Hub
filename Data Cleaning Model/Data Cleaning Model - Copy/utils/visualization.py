import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def serialize_plot(fig):
    """Helper function to properly serialize Plotly figures."""
    try:
        return {
            'data': fig.to_dict()['data'],
            'layout': fig.to_dict()['layout']
        }
    except Exception as e:
        raise Exception(f"Error serializing plot: {str(e)}")

def create_correlation_matrix(df):
    """Create a correlation matrix."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().round(4)
    
    fig = px.imshow(corr_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    aspect="auto")
    
    fig.update_layout(
        width=900,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        title={
            'text': "Correlation Matrix",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return serialize_plot(fig)

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create scatter plot data."""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                    title=f"Scatter Plot: {x_col} vs {y_col}")
    
    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
        template="plotly_white",
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return serialize_plot(fig)

def perform_pca_visualization(df):
    """Perform PCA and return visualization data."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for PCA")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    
    fig = px.scatter(x=components[:, 0], y=components[:, 1],
                    labels={'x': 'First Principal Component',
                           'y': 'Second Principal Component'},
                    title='PCA Visualization')
    
    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return serialize_plot(fig)

def detect_anomalies(df, column):
    """Detect and visualize anomalies using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
    
    fig = px.box(df, y=column, title=f"Anomaly Detection for {column}")
    fig.add_scatter(x=df.index[outlier_mask], y=df[column][outlier_mask],
                   mode='markers', name='Outliers', marker=dict(color='red'))
    return fig

def create_time_series(df, time_column, value_column):
    """Create time series visualization."""
    fig = px.line(df, x=time_column, y=value_column,
                  title=f"Time Series: {value_column} over {time_column}")
    return fig

def create_missing_data_matrix(df):
    """Create missing data visualization."""
    missing = df.isnull()
    fig = px.imshow(missing,
                    labels=dict(color="Missing"),
                    title="Missing Data Matrix",
                    aspect='auto')
    
    fig.update_layout(
        width=1000,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return serialize_plot(fig)

def create_cluster_visualization(df, columns, n_clusters=3):
    """Create cluster visualization using K-means."""
    data = df[columns]
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    
    if len(columns) >= 2:
        fig = px.scatter(data, x=columns[0], y=columns[1],
                        color=clusters, title="Cluster Visualization")
    else:
        fig = px.scatter(data, x=columns[0], color=clusters,
                        title="Cluster Visualization")
    return fig

def create_distribution_plot(df, column):
    """Create distribution plot for a numeric column."""
    try:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[column], name='Distribution',
                                 nbinsx=30, histnorm='probability'))
        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Frequency',
            template='plotly_dark',
            height=500,
            width=700
        )
        return serialize_plot(fig)
    except Exception as e:
        raise Exception(f"Error creating distribution plot: {str(e)}")

def create_summary_dashboard(df):
    """Create a dashboard with multiple plots."""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Missing Values', 'Data Types', 'Value Counts', 'Numeric Distribution'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Missing values plot
        missing = df.isnull().sum()
        fig.add_trace(
            go.Bar(x=missing.index, y=missing.values, name='Missing Values'),
            row=1, col=1
        )

        # Data types plot
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), 
                  values=dtype_counts.values, name='Data Types'),
            row=1, col=2
        )

        # Numeric columns distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Box(x=df[numeric_cols[0]], name=numeric_cols[0]),
                row=2, col=1
            )

        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            template='plotly_white',
            title={
                'text': "Data Summary Dashboard",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return serialize_plot(fig)
    except Exception as e:
        raise Exception(f"Error creating dashboard: {str(e)}")

# Helper function to create consistent layout settings
def apply_default_layout(fig, title, width=900, height=600):
    """Apply consistent layout settings to a plotly figure."""
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    return fig
