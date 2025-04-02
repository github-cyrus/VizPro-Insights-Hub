from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append('./Useful insights predicition model')
sys.path.append('./Data Cleaning Model')

# Import your existing Python tools
from Useful_insights_prediction_model.app import predict_insights  # Update with actual function name
from Data_Cleaning_Model.app import clean_data  # Update with actual function name

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for insights prediction tool
@app.route('/insights-prediction', methods=['GET', 'POST'])
def insights_prediction():
    if request.method == 'GET':
        return render_template('insights_prediction.html')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Call your insights prediction function
                results = predict_insights(filepath)
                return jsonify({'results': results})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return jsonify({'error': 'Invalid file type'}), 400

# Route for data cleaning tool
@app.route('/data-cleaning', methods=['GET', 'POST'])
def data_cleaning():
    if request.method == 'GET':
        return render_template('data_cleaning.html')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Call your data cleaning function
                cleaned_data = clean_data(filepath)
                return jsonify({'cleaned_data': cleaned_data})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 