from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from io import BytesIO
from utils.data_analysis import analyze_columns
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'})

        df = read_file(file)
        if df is None:
            return jsonify({'success': False, 'error': 'Unsupported file format'})

        # Clean and prepare data
        preview_data = clean_data_for_json(df.head())
        
        # Ensure analysis is not None
        analysis = analyze_columns(df) or {
            'data_types': {},
            'missing_values': {},
            'unique_counts': {},
            'numeric_columns': [],
            'categorical_columns': [],
            'column_stats': {}
        }
        
        if not preview_data:
            return jsonify({'success': False, 'error': 'Error processing data'})

        return jsonify({
            'success': True,
            'preview': preview_data,
            'columns': df.columns.tolist(),
            'analysis': analysis,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        })
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def read_file(file):
    try:
        if file.filename.endswith('.csv'):
            return pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        return None
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return None

def clean_data_for_json(df):
    """Clean DataFrame to ensure JSON serialization."""
    try:
        # Replace problematic values
        df_clean = df.replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        })
        
        # Convert to dictionary carefully
        records = []
        for _, row in df_clean.iterrows():
            clean_row = {}
            for column in df_clean.columns:
                value = row[column]
                if pd.isna(value):
                    clean_row[column] = None
                else:
                    clean_row[column] = str(value) if isinstance(value, (np.generic, pd.Timestamp)) else value
            records.append(clean_row)
        
        return records
    except Exception as e:
        logging.error(f"Error in clean_data_for_json: {str(e)}")
        return []

@app.route('/clean', methods=['POST'])
def clean_data():
    try:
        data = request.json
        if not data or 'data' not in data or 'operations' not in data:
            return jsonify({'error': 'Invalid request data'}), 400

        # Convert to DataFrame and handle NaN values
        df = pd.DataFrame(data['data']).replace(['NaN', 'null', ''], np.nan)
        operations = data['operations']
        
        # Track failed operations
        failed_operations = []
        
        for op in operations:
            try:
                # Validate operation before processing
                if not all(key in op for key in ['type', 'column']):
                    failed_operations.append({
                        'type': op.get('type', 'unknown'),
                        'error': 'Missing required parameters'
                    })
                    continue

                # Validate column exists
                if op['column'] not in df.columns:
                    failed_operations.append({
                        'type': op['type'],
                        'column': op['column'],
                        'error': 'Column not found'
                    })
                    continue

                if op['type'] == 'fill_mean':
                    df[op['column']] = pd.to_numeric(df[op['column']], errors='coerce')
                    mean_value = df[op['column']].mean()
                    if pd.isna(mean_value):
                        raise ValueError("Cannot calculate mean of non-numeric data")
                    df[op['column']].fillna(mean_value, inplace=True)
                elif op['type'] == 'fill_median':
                    df[op['column']] = pd.to_numeric(df[op['column']], errors='coerce')
                    df[op['column']].fillna(df[op['column']].median(), inplace=True)
                elif op['type'] == 'fill_mode':
                    df[op['column']].fillna(df[op['column']].mode()[0], inplace=True)
                elif op['type'] == 'fill_value':
                    df[op['column']].fillna(op['value'], inplace=True)
                elif op['type'] == 'remove_rows':
                    df.dropna(subset=[op['column']], inplace=True)
                elif op['type'] == 'ffill':
                    df[op['column']].fillna(method='ffill', inplace=True)
                elif op['type'] == 'bfill':
                    df[op['column']].fillna(method='bfill', inplace=True)
                elif op['type'] == 'interpolate':
                    df[op['column']] = pd.to_numeric(df[op['column']], errors='coerce')
                    df[op['column']].interpolate(method='linear', inplace=True)
            except Exception as e:
                failed_operations.append({
                    'type': op['type'],
                    'column': op['column'],
                    'error': str(e)
                })
                logging.error(f"Error in operation {op['type']}: {str(e)}")
                continue

        # Clean data for JSON response
        cleaned_preview = clean_data_for_json(df.head())
        analysis = analyze_columns(df)

        return jsonify({
            'success': True,
            'preview': cleaned_preview,
            'analysis': analysis,
            'missing_data': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'failed_operations': failed_operations
        })
        
    except Exception as e:
        logging.error(f"Error in clean_data: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/download', methods=['POST'])
def download():
    try:
        data = request.json
        df = pd.DataFrame(data['data'])
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cleaned_data.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

from utils.visualization import (create_correlation_matrix, create_scatter_plot,
                               perform_pca_visualization, detect_anomalies,
                               create_time_series, create_missing_data_matrix,
                               create_cluster_visualization)
from utils.json_utils import serialize_numpy
import json

@app.route('/visualize', methods=['POST'])
def visualize_data():
    try:
        data = request.json
        if not data or 'data' not in data or 'type' not in data:
            return jsonify({'error': 'Invalid request data'}), 400

        df = pd.DataFrame(data['data'])
        viz_type = data['type']
        
        result = None
        error = None

        try:
            if viz_type == 'correlation':
                result = create_correlation_matrix(df)
            elif viz_type == 'scatter':
                x_col = data.get('x_column')
                y_col = data.get('y_column')
                if not x_col or not y_col:
                    raise ValueError("Both x and y columns must be specified")
                result = create_scatter_plot(df, x_col, y_col, data.get('color_column'))
            elif viz_type == 'pca':
                result = perform_pca_visualization(df)
            elif viz_type == 'anomalies':
                column = data.get('column')
                result = detect_anomalies(df, column)
            elif viz_type == 'timeseries':
                time_col = data.get('time_column')
                value_col = data.get('value_column')
                result = create_time_series(df, time_col, value_col)
            elif viz_type == 'missing_matrix':
                result = create_missing_data_matrix(df)
            elif viz_type == 'cluster':
                columns = data.get('columns', [])
                n_clusters = data.get('n_clusters', 3)
                result = create_cluster_visualization(df, columns, n_clusters)
            else:
                raise ValueError('Unsupported visualization type')

            # Serialize numpy arrays and other objects
            serialized_result = serialize_numpy(result)
            
        except Exception as e:
            error = str(e)
            logging.error(f"Visualization error: {error}")

        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400

        if serialized_result is None:
            return jsonify({
                'success': False,
                'error': 'No visualization generated'
            }), 400

        return jsonify({
            'success': True,
            'plot': serialized_result
        })

    except Exception as e:
        logging.error(f"Error in visualization endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/export_report', methods=['POST'])
def export_report():
    try:
        data = request.json
        df = pd.DataFrame(data['data'])
        
        # Generate summary statistics
        summary = {
            'basic_stats': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Create visualizations
        visualizations = {
            'correlation': create_correlation_matrix(df),
            'missing_matrix': create_missing_data_matrix(df)
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'visualizations': {k: v.to_json() for k, v in visualizations.items()}
        })

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
