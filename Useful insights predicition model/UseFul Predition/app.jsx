"use client";  // Add this for Next.js 13+ client components

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';

// Rest of the code remains the same...

const PredictionModelUI = () => {
    const [fileUploaded, setFileUploaded] = useState(false);
    const [targetColumn, setTargetColumn] = useState('');
    const [availableColumns, setAvailableColumns] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [taskType, setTaskType] = useState(null);
    const [trainingComplete, setTrainingComplete] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [alertMessage, setAlertMessage] = useState(null);
    const [alertType, setAlertType] = useState('info');
  
    const regressionModels = ['linear', 'decision_tree', 'random_forest', 'svr', 'knn', 'xgboost'];
    const classificationModels = ['logistic', 'decision_tree', 'random_forest', 'svc', 'knn', 'xgboost'];
  
    const handleFileUpload = (e) => {
      // In a real app, this would process the CSV file
      // For demo purposes, we'll simulate loading data
      setTimeout(() => {
        setFileUploaded(true);
        setAvailableColumns(['age', 'income', 'education', 'employment', 'target']);
        setAlertMessage('Data successfully loaded! Please select your target column.');
        setAlertType('success');
      }, 1000);
    };
  
    const handleTargetSelection = (value) => {
      setTargetColumn(value);
      // In a real app, this would analyze the target column
      // For demo purposes, we'll randomly select a task type
      const detectedTaskType = Math.random() > 0.5 ? 'regression' : 'classification';
      setTaskType(detectedTaskType);
      setAlertMessage(`Target column selected! Detected task type: ${detectedTaskType}`);
      setAlertType('info');
    };
  
    const handleModelSelection = (value) => {
      setSelectedModel(value);
    };
  
    const handleTrainModel = () => {
      setAlertMessage('Training in progress...');
      setAlertType('info');
      
      // Simulate training delay
      setTimeout(() => {
        setTrainingComplete(true);
        setAlertMessage('Model training complete! You can now view results or make predictions.');
        setAlertType('success');
      }, 2000);
    };
  
    const handleViewResults = () => {
      setShowResults(true);
    };
  
    return (
      <div className="max-w-4xl mx-auto p-4">
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Insight Prediction Model</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Step 1: Upload Data */}
              <div>
                <h3 className="text-lg font-medium mb-2">Step 1: Upload your data</h3>
                <div className="flex items-center space-x-4">
                  <input
                    type="file"
                    accept=".csv"
                    className="block w-full text-sm text-slate-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-50 file:text-blue-700
                    hover:file:bg-blue-100"
                    onChange={handleFileUpload}
                  />
                  {fileUploaded && <CheckCircle2 className="text-green-500" size={20} />}
                </div>
              </div>
  
              {/* Step 2: Select Target Column */}
              {fileUploaded && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Step 2: Select target column</h3>
                  <Select onValueChange={handleTargetSelection} value={targetColumn}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select target column" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableColumns.map((column) => (
                        <SelectItem key={column} value={column}>
                          {column}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
  
              {/* Step 3: Select Model */}
              {taskType && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Step 3: Select model</h3>
                  <Select onValueChange={handleModelSelection} value={selectedModel}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      {taskType === 'regression' 
                        ? regressionModels.map(model => (
                            <SelectItem key={model} value={model}>
                              {model.charAt(0).toUpperCase() + model.slice(1).replace('_', ' ')}
                            </SelectItem>
                          ))
                        : classificationModels.map(model => (
                            <SelectItem key={model} value={model}>
                              {model.charAt(0).toUpperCase() + model.slice(1).replace('_', ' ')}
                            </SelectItem>
                          ))
                      }
                    </SelectContent>
                  </Select>
                </div>
              )}
  
              {/* Step 4: Train Model */}
              {selectedModel && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Step 4: Train model</h3>
                  <Button onClick={handleTrainModel} className="w-full">
                    Train Model
                  </Button>
                </div>
              )}
  
              {/* Step 5: Results & Prediction */}
              {trainingComplete && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Step 5: Results & Prediction</h3>
                  <div className="flex space-x-4">
                    <Button onClick={handleViewResults} variant="outline" className="flex-1">
                      View Results
                    </Button>
                    <Button className="flex-1">
                      Make Predictions
                    </Button>
                  </div>
                </div>
              )}
  
              {/* Alert Message */}
              {alertMessage && (
                <Alert variant={alertType === 'success' ? 'default' : 'destructive'}>
                  {alertType === 'success' ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <AlertCircle className="h-4 w-4" />
                  )}
                  <AlertTitle>
                    {alertType === 'success' ? 'Success' : 'Information'}
                  </AlertTitle>
                  <AlertDescription>
                    {alertMessage}
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
        </Card>
  
        {/* Results Section */}
        {showResults && (
          <Card>
            <CardHeader>
              <CardTitle>Model Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {taskType === 'regression' ? (
                  <>
                    <div>
                      <h4 className="font-medium">Performance Metrics</h4>
                      <p>R² Score: 0.87</p>
                      <p>Mean Absolute Error: 0.245</p>
                      <p>Root Mean Squared Error: 0.312</p>
                    </div>
                    <div>
                      <h4 className="font-medium">Visualization</h4>
                      <p className="text-muted-foreground">
                        [This is where actual visualization charts would appear in a real application]
                      </p>
                      <div className="h-64 bg-slate-100 rounded-md flex items-center justify-center">
                        Actual vs Predicted Values Plot
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <h4 className="font-medium">Performance Metrics</h4>
                      <p>Accuracy: 0.92</p>
                      <p>Precision: 0.89</p>
                      <p>Recall: 0.87</p>
                      <p>F1 Score: 0.88</p>
                    </div>
                    <div>
                      <h4 className="font-medium">Visualization</h4>
                      <p className="text-muted-foreground">
                        [This is where actual visualization charts would appear in a real application]
                      </p>
                      <div className="h-64 bg-slate-100 rounded-md flex items-center justify-center">
                        Confusion Matrix
                      </div>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  };
  
  export default PredictionModelUI;