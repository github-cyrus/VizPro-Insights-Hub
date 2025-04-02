import subprocess
import sys
import os
from threading import Thread

def run_insights_prediction():
    insights_dir = os.path.join(os.getcwd(), "Useful insights predicition model")
    if os.path.exists(insights_dir):
        os.chdir(insights_dir)
        os.environ['FLASK_RUN_PORT'] = '5000'
        subprocess.run([sys.executable, "app.py"])
    else:
        print(f"Error: Directory not found - {insights_dir}")

def run_data_cleaning():
    cleaning_dir = os.path.join(os.getcwd(), "Data Cleaning Model", "Data Cleaning Model")
    if os.path.exists(cleaning_dir):
        os.chdir(cleaning_dir)
        os.environ['FLASK_RUN_PORT'] = '5001'
        subprocess.run([sys.executable, "app.py"])
    else:
        print(f"Error: Directory not found - {cleaning_dir}")
        # Try alternate path
        cleaning_dir = os.path.join(os.getcwd(), "Data Cleaning Model")
        if os.path.exists(cleaning_dir):
            os.chdir(cleaning_dir)
            subprocess.run([sys.executable, "app.py"])
        else:
            print(f"Error: Directory not found - {cleaning_dir}")

def run_insights_visualizations():
    viz_dir = os.path.join(os.getcwd(), "Insights and visulizations")
    if os.path.exists(viz_dir):
        os.chdir(viz_dir)
        os.environ['FLASK_RUN_PORT'] = '5002'
        subprocess.run([sys.executable, "app.py"])
    else:
        print(f"Error: Directory not found - {viz_dir}")

if __name__ == "__main__":
    # Store the original directory
    original_dir = os.getcwd()
    
    # Create threads for each application
    insights_thread = Thread(target=run_insights_prediction)
    cleaning_thread = Thread(target=run_data_cleaning)
    viz_thread = Thread(target=run_insights_visualizations)
    
    # Start all applications
    insights_thread.start()
    
    # Return to original directory before starting second app
    os.chdir(original_dir)
    cleaning_thread.start()
    
    # Return to original directory before starting third app
    os.chdir(original_dir)
    viz_thread.start()
    
    # Wait for all to complete (they'll run indefinitely)
    insights_thread.join()
    cleaning_thread.join()
    viz_thread.join() 