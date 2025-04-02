@echo off
echo Starting VizPro Insights Hub servers...

:: Start the Insights Prediction Model
start cmd /k "cd "Useful insights predicition model" && python app.py"
echo Started Insights Prediction Model on port 5000
timeout /t 5

:: Start the Data Cleaning Model
start cmd /k "cd "Data Cleaning Model\Data Cleaning Model" && set FLASK_APP=app.py && set FLASK_ENV=development && python -m flask run --host=127.0.0.1 --port=5001"
echo Started Data Cleaning Model on port 5001
timeout /t 5

:: Start the Insights and Visualizations Model
start cmd /k "cd "Insights and visulizations" && python app.py"
echo Started Insights and Visualizations Model on port 5002

echo.
echo All servers are starting up...
echo.
echo You can access the applications at:
echo - Insights Prediction Model: http://localhost:5000
echo - Data Cleaning Model: http://localhost:5001
echo - Insights and Visualizations: http://localhost:5002
echo.
echo Press any key to exit this window...
pause > nul 