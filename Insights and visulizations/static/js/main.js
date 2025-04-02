// Global variables
let currentData = null;
let currentModel = null;

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const previewTable = document.getElementById('previewTable');
    const insightsContainer = document.getElementById('insightsContainer');
    const visualizationsContainer = document.getElementById('visualizationsContainer');
    
    if (fileInput && uploadButton) {
        uploadButton.addEventListener('click', handleFileUpload);
    }
    
    // Add event listeners for visualization and insights buttons
    const visualizeButton = document.getElementById('visualizeButton');
    const insightsButton = document.getElementById('insightsButton');
    
    if (visualizeButton) {
        visualizeButton.addEventListener('click', handleVisualization);
    }
    
    if (insightsButton) {
        insightsButton.addEventListener('click', handleInsights);
    }
});

// File upload handling
async function handleFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file first');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Update preview table
        updatePreviewTable(data);
        
        // Store the data
        currentData = data;
        
        showSuccess('File uploaded successfully');
    } catch (error) {
        showError('Error uploading file: ' + error.message);
    }
}

// Update preview table
function updatePreviewTable(data) {
    const previewTable = document.getElementById('previewTable');
    if (!previewTable) return;
    
    // Get preview data
    fetch('/preview')
        .then(response => response.json())
        .then(previewData => {
            if (previewData.error) {
                showError(previewData.error);
                return;
            }
            
            // Create table header
            let tableHTML = '<thead><tr>';
            for (const column of data.column_names) {
                tableHTML += `<th>${column}</th>`;
            }
            tableHTML += '</tr></thead>';
            
            // Create table body
            tableHTML += '<tbody>';
            for (const row of previewData) {
                tableHTML += '<tr>';
                for (const column of data.column_names) {
                    tableHTML += `<td>${row[column]}</td>`;
                }
                tableHTML += '</tr>';
            }
            tableHTML += '</tbody>';
            
            previewTable.innerHTML = tableHTML;
        })
        .catch(error => showError('Error loading preview: ' + error.message));
}

// Handle visualization
async function handleVisualization() {
    const visualizationsContainer = document.getElementById('visualizationsContainer');
    if (!visualizationsContainer) return;
    
    try {
        const response = await fetch('/visualize');
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Clear previous visualizations
        visualizationsContainer.innerHTML = '';
        
        // Display distribution plots
        if (data.distribution_plots && data.distribution_plots.length > 0) {
            const distributionSection = document.createElement('div');
            distributionSection.className = 'visualization-section mb-4';
            distributionSection.innerHTML = '<h3>Distribution Plots</h3>';
            
            // Create a container for all distribution plots
            const plotsContainer = document.createElement('div');
            plotsContainer.className = 'visualization-grid';
            
            for (const plot of data.distribution_plots) {
                const plotCard = document.createElement('div');
                plotCard.className = 'visualization-card';
                
                const plotDiv = document.createElement('div');
                plotDiv.className = 'plotly-graph';
                plotDiv.id = `distribution-${plot.name.replace(/\s+/g, '-')}`;
                
                plotCard.appendChild(plotDiv);
                plotsContainer.appendChild(plotCard);
                
                // Wait for the DOM to update before creating the plot
                setTimeout(() => {
                    Plotly.newPlot(plotDiv.id, plot.plot.data, plot.plot.layout);
                }, 0);
            }
            
            distributionSection.appendChild(plotsContainer);
            visualizationsContainer.appendChild(distributionSection);
        }
        
        // Display correlation matrix
        if (data.correlation_matrix) {
            const correlationSection = document.createElement('div');
            correlationSection.className = 'visualization-section mb-4';
            correlationSection.innerHTML = '<h3>Correlation Matrix</h3>';
            
            const plotCard = document.createElement('div');
            plotCard.className = 'visualization-card';
            
            const plotDiv = document.createElement('div');
            plotDiv.className = 'plotly-graph';
            plotDiv.id = 'correlation-matrix';
            
            plotCard.appendChild(plotDiv);
            correlationSection.appendChild(plotCard);
            
            // Wait for the DOM to update before creating the plot
            setTimeout(() => {
                Plotly.newPlot(plotDiv.id, data.correlation_matrix.data, data.correlation_matrix.layout);
            }, 0);
            
            visualizationsContainer.appendChild(correlationSection);
        }
        
        // Display scatter plots
        if (data.scatter_plots && data.scatter_plots.length > 0) {
            const scatterSection = document.createElement('div');
            scatterSection.className = 'visualization-section mb-4';
            scatterSection.innerHTML = '<h3>Scatter Plots</h3>';
            
            // Create a container for all scatter plots
            const plotsContainer = document.createElement('div');
            plotsContainer.className = 'visualization-grid';
            
            for (const plot of data.scatter_plots) {
                const plotCard = document.createElement('div');
                plotCard.className = 'visualization-card';
                
                const plotDiv = document.createElement('div');
                plotDiv.className = 'plotly-graph';
                plotDiv.id = `scatter-${plot.name.replace(/\s+/g, '-')}`;
                
                plotCard.appendChild(plotDiv);
                plotsContainer.appendChild(plotCard);
                
                // Wait for the DOM to update before creating the plot
                setTimeout(() => {
                    Plotly.newPlot(plotDiv.id, plot.plot.data, plot.plot.layout);
                }, 0);
            }
            
            scatterSection.appendChild(plotsContainer);
            visualizationsContainer.appendChild(scatterSection);
        }
        
    } catch (error) {
        showError('Error generating visualizations: ' + error.message);
    }
}

// Handle insights
async function handleInsights() {
    const insightsContainer = document.getElementById('insightsContainer');
    if (!insightsContainer) return;
    
    try {
        const response = await fetch('/get_insights');
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Clear previous insights
        insightsContainer.innerHTML = '';
        
        // Display dataset summary
        if (data.dataset_summary && data.dataset_summary.length > 0) {
            const summarySection = document.createElement('div');
            summarySection.className = 'insights-section';
            summarySection.innerHTML = '<h3>Dataset Summary</h3><ul>';
            data.dataset_summary.forEach(summary => {
                summarySection.innerHTML += `<li>${summary}</li>`;
            });
            summarySection.innerHTML += '</ul>';
            insightsContainer.appendChild(summarySection);
        }
        
        // Display column insights
        if (data.column_insights) {
            const columnSection = document.createElement('div');
            columnSection.className = 'insights-section';
            columnSection.innerHTML = '<h3>Column Insights</h3>';
            
            for (const [column, insights] of Object.entries(data.column_insights)) {
                const columnDiv = document.createElement('div');
                columnDiv.className = 'column-insights';
                columnDiv.innerHTML = `<h4>${column}</h4><ul>`;
                insights.forEach(insight => {
                    columnDiv.innerHTML += `<li>${insight}</li>`;
                });
                columnDiv.innerHTML += '</ul>';
                columnSection.appendChild(columnDiv);
            }
            
            insightsContainer.appendChild(columnSection);
        }
        
        // Display relationships
        if (data.relationships && data.relationships.length > 0) {
            const relationshipSection = document.createElement('div');
            relationshipSection.className = 'insights-section';
            relationshipSection.innerHTML = '<h3>Relationships</h3><ul>';
            data.relationships.forEach(rel => {
                relationshipSection.innerHTML += `<li>${rel.message}</li>`;
            });
            relationshipSection.innerHTML += '</ul>';
            insightsContainer.appendChild(relationshipSection);
        }
        
        // Display recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            const recommendationSection = document.createElement('div');
            recommendationSection.className = 'insights-section';
            recommendationSection.innerHTML = '<h3>Recommendations</h3><ul>';
            data.recommendations.forEach(rec => {
                recommendationSection.innerHTML += `<li>${rec}</li>`;
            });
            recommendationSection.innerHTML += '</ul>';
            insightsContainer.appendChild(recommendationSection);
        }
        
    } catch (error) {
        showError('Error generating insights: ' + error.message);
    }
}

// Utility functions
function showError(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.insertBefore(alertDiv, document.body.firstChild);
}

function showSuccess(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.insertBefore(alertDiv, document.body.firstChild);
} 