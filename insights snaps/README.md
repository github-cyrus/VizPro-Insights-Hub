# InsightSnap 📊

InsightSnap is a powerful data visualization tool that automatically generates visual insights from your data using natural language prompts.

## Features

- 📤 Drag-and-drop file upload for CSV and Excel files
- 🤖 Automatic insight generation
- 💬 Natural language processing for custom visualizations
- 📊 Interactive charts and graphs
- 📱 Responsive design

## Setup Instructions

1. Make sure you have Python installed (Python 3.7 or higher)

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Using InsightSnap

1. Upload your data file (CSV or Excel) using the upload button or drag-and-drop interface
2. Choose between:
   - **Auto-Generate Insights**: Automatically analyzes your data and creates relevant visualizations
   - **Custom Visualization**: Enter natural language prompts to create specific visualizations

## Example Prompts

- "Show me a bar chart of sales by region"
- "Display the distribution of ages"
- "Create a scatter plot showing the relationship between price and quantity"
- "Show the trend of revenue over time"

## Supported File Types

- CSV (.csv)
- Excel (.xlsx, .xls)

## Requirements

See `requirements.txt` for a full list of dependencies. 