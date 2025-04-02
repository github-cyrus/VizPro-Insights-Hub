# ML Model Builder with Data Insights

This is a web application that allows users to upload data, analyze it, train machine learning models, and visualize insights.

## Features

1. Data Upload
   - Upload cleaned CSV files
   - Automatic data type detection
   - Basic statistics generation

2. Data Analysis
   - Preview of the first few rows
   - Column type identification (numerical/categorical)
   - Missing value analysis
   - Basic statistical summaries

3. Model Training
   - Support for multiple ML models:
     - Linear Regression
     - Logistic Regression
     - Random Forest Classifier
     - Random Forest Regressor
   - Automatic feature preprocessing
   - Model performance evaluation

4. Visualization
   - Distribution plots for numerical columns
   - Correlation matrix
   - Feature importance plots (for applicable models)

## Setup and Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Click "Choose File" and select your cleaned CSV file
2. Review the data preview and analysis
3. Select your target column and preferred machine learning model
4. Train the model and view the results
5. Explore the visualizations and insights generated from your data

## Requirements

- Python 3.7+
- Modern web browser
- CSV data file (cleaned)

## File Structure

- `app.py`: Main Flask application
- `templates/index.html`: Frontend HTML template
- `static/main.js`: Frontend JavaScript code
- `requirements.txt`: Python package dependencies
- `uploads/`: Directory for temporary file storage
