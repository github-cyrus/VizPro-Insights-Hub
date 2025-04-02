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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == 'logistic_regression':
            model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        elif model_type == 'random_forest_classifier':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return jsonify({'error': 'Please select logistic_regression or random_forest_classifier'}), 400
        
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = float(model.score(X_train, y_train))
        test_accuracy = float(model.score(X_test, y_test))
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{model_type}_{timestamp}.pkl'
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': X.columns.tolist(),
                'target_column': target_column,
                'model_type': model_type
            }, f)
        
        # Return results
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'model_filename': model_filename,
            'download_url': f'/download_model/{model_filename}',
            'metrics': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download_model/<filename>')
def download_model(filename):
    try:
        if not filename or not os.path.exists(os.path.join(app.config['MODELS_FOLDER'], filename)):
            return jsonify({'error': 'Model not found'}), 404
        
        return send_file(
            os.path.join(app.config['MODELS_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['GET'])
def visualize_data():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    insights = []
    
    # Numerical columns distribution
    for col in current_data.select_dtypes(include=[np.number]).columns:
        fig = px.histogram(current_data, x=col, title=f'Distribution of {col}')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'}
        )
        insights.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        })
    
    # Correlation matrix
    correlation = current_data.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(correlation, 
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    insights.append({
        'type': 'heatmap',
        'title': 'Correlation Matrix',
        'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    })
    
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
