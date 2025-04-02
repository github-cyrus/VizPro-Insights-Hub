from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px
import plotly.utils
import json
import os
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
from scipy import stats

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Global variables to store data
current_data = None
trained_model = None
model_filename = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and analyze the data
        current_data = pd.read_csv(filepath)
        
        # Generate basic statistics
        stats = {
            'rows': len(current_data),
            'columns': len(current_data.columns),
            'column_names': current_data.columns.tolist(),
            'dtypes': current_data.dtypes.astype(str).to_dict(),
            'missing_values': current_data.isnull().sum().to_dict(),
            'numeric_columns': current_data.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return jsonify(stats)
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/preview', methods=['GET'])
def preview_data():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    preview = current_data.head(5).to_dict(orient='records')
    return jsonify(preview)

@app.route('/analyze', methods=['GET'])
def analyze_data():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    # Enhanced data analysis
    analysis = {
        'numerical_columns': current_data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': current_data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': current_data.isnull().sum().to_dict(),
        'summary_stats': current_data.describe().to_dict(),
        'correlation_matrix': current_data.select_dtypes(include=[np.number]).corr().to_dict(),
        'unique_values': {col: current_data[col].nunique() for col in current_data.columns}
    }
    
    return jsonify(analysis)

@app.route('/train', methods=['POST'])
def train_model():
    global trained_model, model_filename, current_data
    
    try:
        # Get user inputs
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        target_column = data.get('target_column')
        model_type = data.get('model_type')
        
        # Basic validation
        if not target_column or not model_type:
            return jsonify({'error': 'Please select target column and model type'}), 400
            
        if current_data is None:
            return jsonify({'error': 'Please upload data first'}), 400
            
        if target_column not in current_data.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in dataset'}), 400
        
        # Prepare data
        X = current_data.drop(columns=[target_column])
        y = current_data[target_column]
        
        # Handle categorical variables
        X = pd.get_dummies(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Calculate split sizes
        split_info = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_percentage': 80,
            'test_percentage': 20
        }
        
        # Train model
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'logistic_regression':
            model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        elif model_type == 'random_forest_classifier':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400
        
        # Train the model
        model.fit(X_train, y_train)
        trained_model = model
        
        # Calculate predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate performance metrics
        train_score = float(model.score(X_train, y_train))
        test_score = float(model.score(X_test, y_test))
        
        metrics = {
            'train_score': train_score,
            'test_score': test_score,
            'split_info': split_info
        }
        
        # Add specific metrics based on model type
        if model_type in ['linear_regression', 'random_forest_regressor']:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics.update({
                'train_mse': float(mean_squared_error(y_train, y_train_pred)),
                'test_mse': float(mean_squared_error(y_test, y_test_pred)),
                'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
                'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
                'train_r2': float(r2_score(y_train, y_train_pred)),
                'test_r2': float(r2_score(y_test, y_test_pred))
            })
        else:  # Classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            try:
                metrics.update({
                    'train_precision': float(precision_score(y_train, y_train_pred, average='weighted')),
                    'test_precision': float(precision_score(y_test, y_test_pred, average='weighted')),
                    'train_recall': float(recall_score(y_train, y_train_pred, average='weighted')),
                    'test_recall': float(recall_score(y_test, y_test_pred, average='weighted')),
                    'train_f1': float(f1_score(y_train, y_train_pred, average='weighted')),
                    'test_f1': float(f1_score(y_test, y_test_pred, average='weighted'))
                })
            except Exception as e:
                print(f"Warning: Could not calculate classification metrics: {str(e)}")
                metrics.update({
                    'train_accuracy': train_score,
                    'test_accuracy': test_score
                })
        
        # Get feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                feature_importance = dict(zip(X.columns, abs(model.coef_).tolist()))
            else:
                feature_importance = dict(zip(X.columns, abs(model.coef_).mean(axis=0).tolist()))
        
        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{model_type}_{timestamp}.pkl'
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': X.columns.tolist(),
                'target_column': target_column,
                'model_type': model_type,
                'split_info': split_info
            }, f)
        
        return jsonify({
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_filename': model_filename
        })
        
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        return jsonify({
            'error': f'Error training model: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/download_model/<filename>')
def download_model(filename):
    if not filename or not os.path.exists(os.path.join(app.config['MODELS_FOLDER'], filename)):
        return jsonify({'error': 'Model not found'})
    
    return send_file(
        os.path.join(app.config['MODELS_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/visualize', methods=['GET'])
def visualize_data():
    try:
        if current_data is None:
            return jsonify({'error': 'No data uploaded'})
        
        # Initialize empty visualizations dictionary
        visualizations = {
            'distribution_plots': [],
            'correlation_matrix': None,
            'scatter_plots': [],
            'insights': []
        }
        
        # Get numerical columns safely
        numerical_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return jsonify({'error': 'No numerical columns found in the dataset'})
        
        # Create distribution plots
        for col in numerical_cols:
            try:
                # Handle NaN values
                clean_data = current_data[col].dropna()
                if len(clean_data) == 0:
                    continue
                    
                fig = px.histogram(
                    clean_data, 
                    title=f'Distribution of {col}',
                    template='plotly_dark',
                    color_discrete_sequence=['#4facfe']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    font={'color': '#ffffff'}
                )
                visualizations['distribution_plots'].append({
                    'name': col,
                    'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                })
            except Exception as e:
                print(f"Error creating distribution plot for {col}: {str(e)}")
                continue
        
        # Create correlation matrix if we have multiple numerical columns
        if len(numerical_cols) > 1:
            try:
                # Handle NaN values in correlation matrix
                corr_matrix = current_data[numerical_cols].fillna(0).corr()
                fig = px.imshow(
                    corr_matrix,
                    title='Correlation Matrix',
                    template='plotly_dark',
                    color_continuous_scale=['#00f2fe', '#ffffff', '#4facfe']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    font={'color': '#ffffff'}
                )
                visualizations['correlation_matrix'] = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                
                # Generate correlation insights
                np.fill_diagonal(corr_matrix.values, 0)
                strongest_corr = np.abs(corr_matrix).unstack().sort_values(ascending=False)[:5]
                for (col1, col2), corr_value in strongest_corr.items():
                    if col1 != col2:
                        visualizations['insights'].append({
                            'type': 'correlation',
                            'message': f'Strong {("positive" if corr_value > 0 else "negative")} correlation ({corr_value:.2f}) between {col1} and {col2}'
                        })
            except Exception as e:
                print(f"Error creating correlation matrix: {str(e)}")
        
        # Create scatter plots for feature relationships
        if len(numerical_cols) > 1:
            for i, col1 in enumerate(numerical_cols[:-1]):
                try:
                    col2 = numerical_cols[i + 1]
                    # Handle NaN values
                    clean_data = current_data[[col1, col2]].dropna()
                    if len(clean_data) == 0:
                        continue
                        
                    fig = px.scatter(
                        clean_data,
                        x=col1,
                        y=col2,
                        title=f'{col1} vs {col2}',
                        template='plotly_dark',
                        color_discrete_sequence=['#4facfe']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.05)',
                        font={'color': '#ffffff'}
                    )
                    visualizations['scatter_plots'].append({
                        'name': f'{col1} vs {col2}',
                        'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    })
                except Exception as e:
                    print(f"Error creating scatter plot for {col1} vs {col2}: {str(e)}")
                    continue
        
        return jsonify(visualizations)
        
    except Exception as e:
        print(f"Error in visualize_data: {str(e)}")
        return jsonify({
            'error': f'Error generating visualizations: {str(e)}',
            'status': 'error'
        })

def generate_numerical_insights(df, column):
    """Generate insights for numerical columns"""
    insights = []
    data = df[column].dropna()
    
    # Basic statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    
    insights.append(f"The average value of {column} is {mean_val:.2f}")
    insights.append(f"The median value of {column} is {median_val:.2f}")
    insights.append(f"The standard deviation of {column} is {std_val:.2f}")
    
    # Distribution shape
    skewness = data.skew()
    if abs(skewness) > 1:
        direction = "right" if skewness > 0 else "left"
        insights.append(f"The distribution of {column} is {direction}-skewed (skewness: {skewness:.2f})")
    
    # Outliers
    z_scores = stats.zscore(data)
    outliers = np.abs(z_scores) > 3
    if outliers.any():
        outlier_count = outliers.sum()
        insights.append(f"There are {outlier_count} potential outliers in {column}")
    
    return insights

def generate_categorical_insights(df, column):
    """Generate insights for categorical columns"""
    insights = []
    value_counts = df[column].value_counts()
    
    # Most common values
    top_values = value_counts.head(3)
    insights.append(f"The most common values in {column} are: {', '.join([f'{val} ({count})' for val, count in top_values.items()])}")
    
    # Unique values
    unique_count = value_counts.shape[0]
    insights.append(f"There are {unique_count} unique values in {column}")
    
    # Missing values
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        missing_percentage = (missing_count / len(df)) * 100
        insights.append(f"There are {missing_count} missing values ({missing_percentage:.1f}%) in {column}")
    
    return insights

@app.route('/get_insights', methods=['GET'])
def get_data_insights():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        insights = {
            'dataset_summary': [],
            'column_insights': {},
            'relationships': [],
            'recommendations': []
        }
        
        # Dataset summary
        insights['dataset_summary'].extend([
            f"Dataset contains {len(current_data)} rows and {len(current_data.columns)} columns",
            f"Total memory usage: {current_data.memory_usage(deep=True).sum() / 1024:.2f} KB"
        ])
        
        # Column-wise insights
        for column in current_data.columns:
            column_insights = []
            
            if pd.api.types.is_numeric_dtype(current_data[column]):
                column_insights.extend(generate_numerical_insights(current_data, column))
            else:
                column_insights.extend(generate_categorical_insights(current_data, column))
            
            insights['column_insights'][column] = column_insights
        
        # Relationship insights
        numerical_cols = current_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = current_data[numerical_cols].corr()
            np.fill_diagonal(corr_matrix.values, 0)
            
            # Find strong correlations
            strong_correlations = np.abs(corr_matrix).unstack().sort_values(ascending=False)[:5]
            for (col1, col2), corr_value in strong_correlations.items():
                if col1 != col2 and abs(corr_value) > 0.5:
                    insights['relationships'].append({
                        'type': 'correlation',
                        'columns': [col1, col2],
                        'strength': corr_value,
                        'message': f"Strong {('positive' if corr_value > 0 else 'negative')} correlation ({corr_value:.2f}) between {col1} and {col2}"
                    })
        
        # Generate recommendations
        if current_data.isnull().any().any():
            insights['recommendations'].append("Consider handling missing values in the dataset")
        
        if len(current_data) < 100:
            insights['recommendations'].append("Dataset size is relatively small, consider collecting more data")
        
        # Check for potential data quality issues
        for column in current_data.columns:
            if current_data[column].dtype == 'object':
                unique_ratio = current_data[column].nunique() / len(current_data)
                if unique_ratio > 0.9:
                    insights['recommendations'].append(f"Column '{column}' has high cardinality ({unique_ratio:.1%} unique values)")
        
        return jsonify(insights)
        
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return jsonify({
            'error': f'Error generating insights: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
