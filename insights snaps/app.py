import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO
import json
from typing import List, Dict, Tuple
import re

# Set page config
st.set_page_config(
    page_title="InsightSnap - Data Visualization",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .upload-box {
        border: 2px dashed #cccccc;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def analyze_data(df: pd.DataFrame) -> List[Dict]:
    """
    Automatically analyze the dataset and suggest visualizations
    """
    suggestions = []
    
    # Check for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numerical_cols) >= 1:
        # Distribution of numerical values
        for col in numerical_cols[:2]:  # Limit to first 2 numerical columns
            suggestions.append({
                'title': f'Distribution of {col}',
                'type': 'histogram',
                'x': col,
                'description': f'Shows the distribution of {col} values across the dataset.'
            })
    
    if len(categorical_cols) >= 1 and len(numerical_cols) >= 1:
        # Bar chart for categorical vs numerical
        suggestions.append({
            'title': f'Average {numerical_cols[0]} by {categorical_cols[0]}',
            'type': 'bar',
            'x': categorical_cols[0],
            'y': numerical_cols[0],
            'description': f'Compares the average {numerical_cols[0]} across different {categorical_cols[0]} categories.'
        })
    
    return suggestions

def create_visualization(df: pd.DataFrame, viz_type: str, x: str, y: str = None, title: str = '') -> go.Figure:
    """
    Create a visualization based on the type and columns
    """
    if viz_type == 'histogram':
        fig = px.histogram(df, x=x, title=title)
    elif viz_type == 'bar':
        fig = px.bar(df, x=x, y=y, title=title)
    elif viz_type == 'scatter':
        fig = px.scatter(df, x=x, y=y, title=title)
    else:
        fig = px.line(df, x=x, y=y, title=title)
    
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        margin=dict(t=50, l=0, r=0, b=0)
    )
    return fig

def process_prompt(prompt: str, df: pd.DataFrame) -> Dict:
    """
    Process natural language prompt to determine visualization type
    """
    prompt = prompt.lower()
    
    # Simple rule-based processing
    viz_type = 'bar'  # default
    if any(word in prompt for word in ['distribution', 'spread']):
        viz_type = 'histogram'
    elif any(word in prompt for word in ['correlation', 'relationship']):
        viz_type = 'scatter'
    elif any(word in prompt for word in ['trend', 'over time']):
        viz_type = 'line'
    
    # Try to identify columns from the prompt
    columns = df.columns.tolist()
    mentioned_cols = [col for col in columns if col.lower() in prompt]
    
    x = mentioned_cols[0] if mentioned_cols else df.columns[0]
    y = mentioned_cols[1] if len(mentioned_cols) > 1 else (
        df.select_dtypes(include=['int64', 'float64']).columns[0]
        if viz_type != 'histogram' else None
    )
    
    return {
        'type': viz_type,
        'x': x,
        'y': y,
        'title': prompt.capitalize()
    }

def analyze_data_query(df: pd.DataFrame, query: str) -> str:
    """
    Enhanced data analysis function that can handle a wide range of queries
    """
    query = query.lower().strip()
    response = ""
    
    try:
        # Data Overview
        if any(word in query for word in ['summary', 'overview', 'describe', 'tell me about']):
            num_rows, num_cols = df.shape
            dtypes = df.dtypes.value_counts()
            response = f"Dataset Overview:\n"
            response += f"• Total rows: {num_rows}\n"
            response += f"• Total columns: {num_cols}\n"
            response += f"• Column types:\n"
            for dtype, count in dtypes.items():
                response += f"  - {dtype}: {count} columns\n"
            response += f"\nNumerical Summary:\n{df.describe()}\n"
            
        # Column Information
        elif any(word in query for word in ['columns', 'fields', 'variables']):
            cols = df.columns.tolist()
            dtypes = df.dtypes
            response = "Column Information:\n"
            for col in cols:
                unique_vals = df[col].nunique()
                response += f"• {col} (Type: {dtypes[col]}):\n"
                response += f"  - Unique values: {unique_vals}\n"
                if df[col].dtype in ['int64', 'float64']:
                    response += f"  - Range: {df[col].min()} to {df[col].max()}\n"
                
        # Missing Values Analysis
        elif any(word in query for word in ['missing', 'null', 'na', 'empty']):
            missing = df.isnull().sum()
            missing_pct = (df.isnull().sum() / len(df)) * 100
            response = "Missing Values Analysis:\n"
            for col in df.columns:
                if missing[col] > 0:
                    response += f"• {col}: {missing[col]} missing values ({missing_pct[col]:.2f}%)\n"
            if missing.sum() == 0:
                response += "No missing values found in the dataset!\n"
                
        # Statistical Analysis
        elif any(word in query for word in ['statistics', 'stats', 'statistical']):
            num_cols = df.select_dtypes(include=['int64', 'float64'])
            response = "Statistical Analysis:\n"
            for col in num_cols.columns:
                response += f"\n{col}:\n"
                response += f"• Mean: {df[col].mean():.2f}\n"
                response += f"• Median: {df[col].median():.2f}\n"
                response += f"• Std Dev: {df[col].std():.2f}\n"
                response += f"• Skewness: {df[col].skew():.2f}\n"
                
        # Correlation Analysis
        elif any(word in query for word in ['correlation', 'correlate', 'relationship']):
            num_cols = df.select_dtypes(include=['int64', 'float64'])
            if len(num_cols.columns) > 1:
                corr = num_cols.corr()
                response = "Correlation Analysis:\n"
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        col1, col2 = corr.columns[i], corr.columns[j]
                        correlation = corr.iloc[i, j]
                        response += f"• {col1} vs {col2}: {correlation:.3f}\n"
            else:
                response = "Not enough numerical columns for correlation analysis."
                
        # Value Counts / Distribution
        elif any(word in query for word in ['distribution', 'frequency', 'count', 'unique values']):
            col_name = None
            for col in df.columns:
                if col.lower() in query:
                    col_name = col
                    break
            
            if col_name:
                value_counts = df[col_name].value_counts()
                response = f"Distribution of {col_name}:\n"
                for val, count in value_counts.head(10).items():
                    response += f"• {val}: {count} ({(count/len(df))*100:.2f}%)\n"
                if len(value_counts) > 10:
                    response += "... (showing top 10 values)\n"
            else:
                response = "Please specify which column's distribution you'd like to see."
                
        # Group By Analysis
        elif any(word in query for word in ['group', 'grouped by', 'compare']):
            # Try to identify columns for grouping
            cols = df.columns
            group_col = None
            measure_col = None
            
            for col in cols:
                if col.lower() in query:
                    if df[col].dtype == 'object':
                        group_col = col
                    else:
                        measure_col = col
            
            if group_col and measure_col:
                grouped = df.groupby(group_col)[measure_col].agg(['mean', 'count'])
                response = f"Analysis of {measure_col} grouped by {group_col}:\n"
                for idx, row in grouped.iterrows():
                    response += f"• {idx}:\n"
                    response += f"  - Average: {row['mean']:.2f}\n"
                    response += f"  - Count: {row['count']}\n"
            else:
                response = "Please specify a categorical column to group by and a numerical column to analyze."
                
        # Basic Aggregations
        elif any(word in query for word in ['average', 'mean', 'median', 'max', 'min', 'sum']):
            num_cols = df.select_dtypes(include=['int64', 'float64'])
            agg_type = 'mean'  # default
            if 'median' in query:
                agg_type = 'median'
            elif 'max' in query:
                agg_type = 'max'
            elif 'min' in query:
                agg_type = 'min'
            elif 'sum' in query:
                agg_type = 'sum'
                
            response = f"{agg_type.capitalize()} values for numerical columns:\n"
            for col in num_cols.columns:
                val = getattr(df[col], agg_type)()
                response += f"• {col}: {val:.2f}\n"
                
        # Shape and Size
        elif any(word in query for word in ['shape', 'size', 'dimensions', 'how many rows', 'how many columns']):
            rows, cols = df.shape
            response = f"Dataset Dimensions:\n"
            response += f"• Number of rows: {rows}\n"
            response += f"• Number of columns: {cols}\n"
            response += f"• Total cells: {rows * cols}\n"
            
        else:
            response = "I can help you analyze this data. Try asking about:\n"
            response += "• Summary statistics and overview\n"
            response += "• Column information and data types\n"
            response += "• Missing values analysis\n"
            response += "• Statistical analysis and correlations\n"
            response += "• Value distributions and frequencies\n"
            response += "• Grouped analysis and comparisons\n"
            response += "• Basic aggregations (mean, median, max, min, sum)\n"
            response += "• Dataset shape and dimensions\n"
            
    except Exception as e:
        response = f"Error analyzing data: {str(e)}\n"
        response += "Please try rephrasing your question."
        
    return response

# Main app layout
st.title("📊 InsightSnap")
st.markdown("### Transform Your Data into Visual Insights")

# File upload
uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Drag and drop your file here"
)

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        
        # Display data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())
        
        # Tabs for different modes
        tab1, tab2 = st.tabs(["🤖 Auto-Generate Insights", "💬 Chat with Your Data"])
        
        with tab1:
            st.markdown("### Automatically Generated Insights")
            suggestions = analyze_data(df)
            
            for i, suggestion in enumerate(suggestions):
                with st.container():
                    st.subheader(suggestion['title'])
                    fig = create_visualization(
                        df,
                        suggestion['type'],
                        suggestion['x'],
                        suggestion.get('y'),
                        suggestion['title']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(suggestion['description'])
                    st.divider()
        
        with tab2:
            st.markdown("### Chat with Your Data")
            st.markdown("""
                Ask questions about your data and I'll help you analyze it. Try asking:
                - "Give me a summary of the data"
                - "What columns are in the dataset?"
                - "Are there any missing values?"
                - "What's the average of numerical columns?"
                - "Show me the distribution of a specific column"
                - "Compare values between different categories"
            """)
            
            # Chat input
            user_query = st.text_input("Ask a question about your data:", key="data_query")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Generate response
                response = analyze_data_query(df, user_query)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Display chat history
            for message in st.session_state.chat_history:
                message_class = "user-message" if message["role"] == "user" else "assistant-message"
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <p>{message["content"]}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Show placeholder when no file is uploaded
    st.markdown("""
        <div class="upload-box">
            <h3>👆 Upload your data file to get started</h3>
            <p>Support for CSV and Excel files</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by InsightSnap Team") 