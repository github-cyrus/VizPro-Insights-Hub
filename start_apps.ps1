Write-Host "Starting VizPro Insights Hub servers..." -ForegroundColor Green

# Start the Insights Prediction Model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Useful insights predicition model'; python app.py"
Write-Host "Started Insights Prediction Model on port 5000" -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the Data Cleaning Model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Data Cleaning Model\Data Cleaning Model'; python app.py"
Write-Host "Started Data Cleaning Model on port 5001" -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the Insights and Visualizations Model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Insights and visulizations'; python app.py"
Write-Host "Started Insights and Visualizations Model on port 5002" -ForegroundColor Yellow

Write-Host "`nAll servers are starting up..." -ForegroundColor Green
Write-Host "`nYou can access the applications at:"
Write-Host "- Insights Prediction Model: http://localhost:5000" -ForegroundColor Cyan
Write-Host "- Data Cleaning Model: http://localhost:5001" -ForegroundColor Cyan
Write-Host "- Insights and Visualizations: http://localhost:5002" -ForegroundColor Cyan
Write-Host "`nPress any key to exit this window..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 