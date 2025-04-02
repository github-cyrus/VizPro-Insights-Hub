document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const modelForm = document.getElementById('modelForm');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('dataFile');
    const loadingOverlay = document.querySelector('.loading-overlay');

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('highlight');
    }

    function unhighlight(e) {
        dropZone.classList.remove('highlight');
    }

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFileUpload();
    }

    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type} fade-in`;
        toast.textContent = message;
        document.getElementById('toastContainer').appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    fileInput.addEventListener('change', () => {
        handleFileUpload();
    });

    window.handleFileUpload = function() {
        if (!fileInput.files.length) {
            showToast('Please select a file first', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        showLoading();
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }
            
            // Show data statistics
            const statsHtml = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <i class="fas fa-table"></i>
                        <h4>Rows</h4>
                        <p>${data.rows}</p>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-columns"></i>
                        <h4>Columns</h4>
                        <p>${data.columns}</p>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-calculator"></i>
                        <h4>Numeric Columns</h4>
                        <p>${data.numeric_columns.length}</p>
                    </div>
                </div>
            `;
            document.querySelector('.data-stats').innerHTML = statsHtml;
            
            // Populate target column dropdown
            const targetSelect = document.getElementById('targetColumn');
            targetSelect.innerHTML = '';
            data.column_names.forEach(column => {
                const option = document.createElement('option');
                option.value = column;
                option.textContent = column;
                targetSelect.appendChild(option);
            });

            // Show sections with animation
            ['previewSection', 'analysisSection', 'trainingSection'].forEach((id, index) => {
                setTimeout(() => {
                    document.getElementById(id).style.display = 'block';
                    document.getElementById(id).classList.add('fade-in');
                }, index * 200);
            });
            
            loadDataPreview();
            loadAnalysis();
            showToast('File uploaded successfully!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error uploading file', 'error');
        })
        .finally(() => {
            hideLoading();
        });
    }

    window.trainModel = function() {
        const targetColumn = document.getElementById('targetColumn').value;
        const modelType = document.getElementById('modelType').value;

        if (!targetColumn) {
            showToast('Please select a target column', 'error');
            return;
        }

        showLoading();

        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_column: targetColumn,
                model_type: modelType
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            // Display training results
            const metrics = data.metrics;
            const isRegression = data.model_type.includes('regressor') || data.model_type === 'linear_regression';
            
            let metricsHtml = `
                <div class="alert alert-success fade-in">
                    <h4><i class="fas fa-chart-line"></i> Model Training Results</h4>
                    
                    <!-- Train-Test Split Information -->
                    <div class="card mb-3">
                        <div class="card-header">
                            <i class="fas fa-random"></i> Train-Test Split
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h5>Training Set</h5>
                                    <p>${metrics.split_info.train_size} samples (${metrics.split_info.train_percentage}%)</p>
                                </div>
                                <div class="col-md-6">
                                    <h5>Test Set</h5>
                                    <p>${metrics.split_info.test_size} samples (${metrics.split_info.test_percentage}%)</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Model Performance Metrics -->
                    <div class="card mb-3">
                        <div class="card-header">
                            <i class="fas fa-chart-bar"></i> Model Performance
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h5>Training Performance</h5>
                                    <ul class="list-unstyled">
                                        ${isRegression ? `
                                            <li>R² Score: ${(metrics.train_r2 * 100).toFixed(2)}%</li>
                                            <li>Mean Squared Error: ${metrics.train_mse.toFixed(4)}</li>
                                            <li>Mean Absolute Error: ${metrics.train_mae.toFixed(4)}</li>
                                        ` : `
                                            <li>Accuracy: ${(metrics.train_score * 100).toFixed(2)}%</li>
                                            ${metrics.train_precision ? `
                                                <li>Precision: ${(metrics.train_precision * 100).toFixed(2)}%</li>
                                                <li>Recall: ${(metrics.train_recall * 100).toFixed(2)}%</li>
                                                <li>F1 Score: ${(metrics.train_f1 * 100).toFixed(2)}%</li>
                                            ` : ''}
                                        `}
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h5>Test Performance</h5>
                                    <ul class="list-unstyled">
                                        ${isRegression ? `
                                            <li>R² Score: ${(metrics.test_r2 * 100).toFixed(2)}%</li>
                                            <li>Mean Squared Error: ${metrics.test_mse.toFixed(4)}</li>
                                            <li>Mean Absolute Error: ${metrics.test_mae.toFixed(4)}</li>
                                        ` : `
                                            <li>Accuracy: ${(metrics.test_score * 100).toFixed(2)}%</li>
                                            ${metrics.test_precision ? `
                                                <li>Precision: ${(metrics.test_precision * 100).toFixed(2)}%</li>
                                                <li>Recall: ${(metrics.test_recall * 100).toFixed(2)}%</li>
                                                <li>F1 Score: ${(metrics.test_f1 * 100).toFixed(2)}%</li>
                                            ` : ''}
                                        `}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    ${data.feature_importance ? `
                        <!-- Feature Importance -->
                        <div class="card">
                            <div class="card-header">
                                <i class="fas fa-list-ol"></i> Top Feature Importance
                            </div>
                            <div class="card-body">
                                <ul class="feature-list">
                                    ${Object.entries(data.feature_importance)
                                        .sort((a, b) => b[1] - a[1])
                                        .slice(0, 5)
                                        .map(([feature, importance]) => `
                                            <li>
                                                <span class="feature-name">${feature}</span>
                                                <div class="progress">
                                                    <div class="progress-bar" style="width: ${(importance * 100).toFixed(2)}%"></div>
                                                </div>
                                                <small class="text-muted">${(importance * 100).toFixed(2)}% importance</small>
                                            </li>
                                        `).join('')}
                                </ul>
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;

            document.getElementById('trainingResults').innerHTML = metricsHtml;

            // Show download button
            const downloadSection = document.getElementById('downloadSection');
            downloadSection.style.display = 'block';
            downloadSection.querySelector('#downloadModel').onclick = () => {
                window.location.href = `/download_model/${data.model_filename}`;
            };

            // Show visualization section and load visualizations
            document.getElementById('visualizationSection').style.display = 'block';
            document.getElementById('visualizationSection').classList.add('fade-in');
            loadVisualizations();
            
            showToast('Model trained successfully!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error training model', 'error');
        })
        .finally(() => {
            hideLoading();
        });
    }

    function loadDataPreview() {
        showLoading();
        fetch('/preview')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            // Create table HTML
            let tableHtml = '<table class="table table-striped table-hover"><thead><tr>';
            const columns = Object.keys(data[0]);
            columns.forEach(column => {
                tableHtml += `<th>${column}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';

            data.forEach(row => {
                tableHtml += '<tr>';
                columns.forEach(column => {
                    tableHtml += `<td>${row[column]}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';

            document.getElementById('dataPreview').innerHTML = tableHtml;
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error loading data preview', 'error');
        })
        .finally(() => {
            hideLoading();
        });
    }

    function loadAnalysis() {
        showLoading();
        fetch('/analyze')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            let analysisHtml = `
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h4><i class="fas fa-calculator"></i> Numerical Columns</h4>
                        <ul class="list-group">
                            ${data.numerical_columns.map(col => `
                                <li class="list-group-item">
                                    ${col}
                                    ${data.missing_values[col] > 0 ? 
                                        `<span class="badge bg-warning">${data.missing_values[col]} missing</span>` : 
                                        '<span class="badge bg-success">Complete</span>'}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    <div class="analysis-card">
                        <h4><i class="fas fa-font"></i> Categorical Columns</h4>
                        <ul class="list-group">
                            ${data.categorical_columns.map(col => `
                                <li class="list-group-item">
                                    ${col}
                                    <span class="badge bg-info">${data.unique_values[col]} unique values</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            `;

            document.getElementById('analysisResults').innerHTML = analysisHtml;
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error loading analysis', 'error');
        })
        .finally(() => {
            hideLoading();
        });
    }

    function loadVisualizations() {
        showLoading();
        fetch('/visualize')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            const visualizationsDiv = document.getElementById('visualizations');
            visualizationsDiv.innerHTML = '';

            // Handle distribution plots
            if (data.distribution_plots && data.distribution_plots.length > 0) {
                const container = document.createElement('div');
                container.className = 'visualization-container fade-in';
                container.dataset.type = 'distribution';
                container.innerHTML = '<h4>Distribution Plots</h4>';
                
                data.distribution_plots.forEach((plot, index) => {
                    const plotDiv = document.createElement('div');
                    plotDiv.id = `dist-plot-${index}`;
                    container.appendChild(plotDiv);
                    
                    setTimeout(() => {
                        Plotly.newPlot(`dist-plot-${index}`, plot.plot.data, plot.plot.layout);
                    }, index * 100);
                });
                
                visualizationsDiv.appendChild(container);
            }

            // Handle correlation matrix
            if (data.correlation_matrix) {
                const container = document.createElement('div');
                container.className = 'visualization-container fade-in';
                container.dataset.type = 'correlation';
                container.innerHTML = '<h4>Correlation Matrix</h4>';
                
                const plotDiv = document.createElement('div');
                plotDiv.id = 'corr-matrix';
                container.appendChild(plotDiv);
                
                setTimeout(() => {
                    Plotly.newPlot('corr-matrix', data.correlation_matrix.data, data.correlation_matrix.layout);
                }, 100);
                
                visualizationsDiv.appendChild(container);
            }

            // Handle scatter plots
            if (data.scatter_plots && data.scatter_plots.length > 0) {
                const container = document.createElement('div');
                container.className = 'visualization-container fade-in';
                container.dataset.type = 'scatter';
                container.innerHTML = '<h4>Feature Relationships</h4>';
                
                data.scatter_plots.forEach((plot, index) => {
                    const plotDiv = document.createElement('div');
                    plotDiv.id = `scatter-plot-${index}`;
                    container.appendChild(plotDiv);
                    
                    setTimeout(() => {
                        Plotly.newPlot(`scatter-plot-${index}`, plot.plot.data, plot.plot.layout);
                    }, index * 100);
                });
                
                visualizationsDiv.appendChild(container);
            }

            // Handle insights
            if (data.insights && data.insights.length > 0) {
                const insightsContainer = document.createElement('div');
                insightsContainer.className = 'insights-container fade-in';
                insightsContainer.innerHTML = '<h4>Key Insights</h4><ul class="insights-list">';
                
                data.insights.forEach(insight => {
                    insightsContainer.innerHTML += `<li>${insight.message}</li>`;
                });
                
                insightsContainer.innerHTML += '</ul>';
                visualizationsDiv.appendChild(insightsContainer);
            }

            // Show the visualization section
            const visualizationSection = document.getElementById('visualizationSection');
            if (visualizationSection) {
                visualizationSection.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error loading visualizations', 'error');
        })
        .finally(() => {
            hideLoading();
        });
    }

    // Chart type filter
    document.getElementById('chartType').addEventListener('change', function(e) {
        const selectedType = e.target.value;
        const containers = document.querySelectorAll('.visualization-container');
        
        containers.forEach(container => {
            if (selectedType === 'all' || container.dataset.type === selectedType) {
                container.style.display = 'block';
            } else {
                container.style.display = 'none';
            }
        });
    });
});
